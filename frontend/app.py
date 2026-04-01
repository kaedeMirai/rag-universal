import requests
import streamlit as st
from settings import settings


def auth_headers() -> dict[str, str]:
    token = st.session_state.get("auth_token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def api_get(path: str, timeout: int = 15):
    return requests.get(f"{settings.backend_url}{path}", headers=auth_headers(), timeout=timeout)


def api_post(path: str, payload: dict | None = None, timeout: int = 30):
    return requests.post(
        f"{settings.backend_url}{path}",
        json=payload,
        headers=auth_headers(),
        timeout=timeout,
    )


def login_view():
    st.title("RAG Ассистент")
    st.caption("Авторизация выполняется через FastAPI и пользователей из Postgres.")

    with st.form("login_form"):
        username = st.text_input("Логин")
        password = st.text_input("Пароль", type="password")
        submitted = st.form_submit_button("Войти")

    if submitted:
        try:
            response = api_post(
                "/auth/login",
                {
                    "username": username,
                    "password": password,
                },
            )
        except Exception as error:
            st.error(f"Не удалось связаться с backend: {error}")
            return

        if response.status_code != 200:
            detail = response.json().get("detail", "Ошибка авторизации")
            st.error(detail)
            return

        data = response.json()
        st.session_state["auth_token"] = data["access_token"]
        st.session_state["current_user"] = data["user"]
        st.rerun()


def logout():
    try:
        api_post("/auth/logout", {})
    except Exception:
        pass

    for key in ["auth_token", "current_user"]:
        st.session_state.pop(key, None)

    st.rerun()


def refresh_current_user():
    try:
        response = api_get("/auth/me")
    except Exception as error:
        st.error(f"Не удалось обновить данные пользователя: {error}")
        return

    if response.status_code == 200:
        st.session_state["current_user"] = response.json()
        return

    st.warning("Сессия истекла, выполните вход заново.")
    logout()


def users_admin_panel():
    user = st.session_state["current_user"]
    if user["role"] != "admin":
        return

    with st.expander("Пользователи", expanded=False):
        if st.button("Обновить список пользователей"):
            st.session_state.pop("users_cache", None)

        if "users_cache" not in st.session_state:
            response = api_get("/auth/users")
            if response.status_code == 200:
                st.session_state["users_cache"] = response.json()
            else:
                st.error(
                    response.json().get("detail", "Не удалось загрузить пользователей")
                )
                return

        st.dataframe(st.session_state["users_cache"], width="stretch")

    with st.expander("Создать пользователя", expanded=False):
        with st.form("create_user_form"):
            username = st.text_input("Логин нового пользователя")
            password = st.text_input("Пароль нового пользователя", type="password")
            first_name = st.text_input("Имя")
            last_name = st.text_input("Фамилия")
            email = st.text_input("Email")
            role = st.selectbox("Роль", ["viewer", "editor", "admin"])
            submitted = st.form_submit_button("Создать")

        if submitted:
            payload = {
                "username": username,
                "password": password,
                "first_name": first_name,
                "last_name": last_name,
                "email": email or None,
                "role": role,
            }
            response = api_post("/auth/users", payload)
            if response.status_code == 201:
                st.success("Пользователь создан")
                st.session_state.pop("users_cache", None)
            else:
                st.error(
                    response.json().get("detail", "Не удалось создать пользователя")
                )


def chat_view():
    user = st.session_state["current_user"]
    st.title(f"RAG Ассистент ({user['first_name']} {user['last_name']})")
    st.caption(f"Роль: {user['role']} | Логин: {user['username']}")

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Проверить backend"):
            try:
                response = api_get("/health", timeout=10)
                if response.status_code == 200:
                    st.success(f"Сервис доступен: {response.json()}")
                else:
                    st.error(f"Ошибка сервиса: {response.status_code}")
            except Exception as error:
                st.error(f"Ошибка запроса: {error}")

    with col2:
        if st.button("Выйти"):
            logout()

    users_admin_panel()

    with st.form("chat_form"):
        query = st.text_input("Ваш вопрос:")
        submitted = st.form_submit_button("Отправить")

    if submitted and query:
        try:
            response = api_post("/chat", {"query": query}, timeout=settings.chat_request_timeout_seconds)
        except Exception as error:
            st.error(f"Ошибка запроса: {error}")
            return

        if response.status_code != 200:
            st.error(response.json().get("detail", "Не удалось получить ответ"))
            return

        data = response.json()
        st.markdown(f"**Вы:** {query}")
        st.markdown(f"**Бот:** {data.get('answer', 'Нет ответа')}")

        confidence = data.get("confidence")
        if confidence is not None:
            st.caption(f"Оценка релевантности контекста: {confidence:.1%}")

        sources = data.get("sources", [])
        if sources:
            st.markdown("**Источники:**")
            for source_path in sources:
                st.code(source_path)
        else:
            st.write("Источники не найдены")


if "auth_token" not in st.session_state:
    login_view()
else:
    if "current_user" not in st.session_state:
        refresh_current_user()

    if "current_user" in st.session_state:
        chat_view()
