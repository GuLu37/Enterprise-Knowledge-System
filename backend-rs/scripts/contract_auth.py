"""双端认证契约测试：同一请求序列打 Python(8000)/Rust(8001) 后端，比对结构并验证令牌互通。

前置：两个后端已启动（Python: uvicorn app.main:app --port 8000 于 backend/；
Rust: cargo run 于 backend-rs/，默认 8001），共用同一 SQLite。

安全约束：连接目标仅为 http.client 显式常量（本机回环测试端口），不构造/不解析任何
动态 URL；管理员口令从 ../backend/.env 的 BOOTSTRAP_ADMIN_PASSWORD 读取；测试用户
口令每次随机生成。只访问本地 HTTP，无其他文件读写。退出码 0 = 全部通过。
"""
import http.client
import json
import uuid

PY = ("127.0.0.1", 8000, "/api/v1")
RS = ("127.0.0.1", 8001, "/api/v1")
ALLOWED_TARGETS = (PY, RS)

failures = []


def load_bootstrap_password():
    """从 backend/.env 读取 BOOTSTRAP_ADMIN_PASSWORD（仅该行）。"""
    for line in open("../../backend/.env", encoding="utf-8"):
        if line.startswith("BOOTSTRAP_ADMIN_PASSWORD="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("backend/.env 缺少 BOOTSTRAP_ADMIN_PASSWORD，无法运行契约测试")


def call(target, method, path, body=None, token=None):
    if target not in ALLOWED_TARGETS:
        raise ValueError("目标不在测试白名单内")
    host, port, prefix = target
    payload = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    conn = http.client.HTTPConnection(host, port, timeout=15)
    try:
        conn.request(method, prefix + path, body=payload, headers=headers)
        resp = conn.getresponse()
        raw = resp.read().decode()
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {}
        return resp.status, parsed
    finally:
        conn.close()


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" | {detail}" if not condition else ""))
    if not condition:
        failures.append(name)


def token_shape(body):
    return (
        isinstance(body.get("access_token"), str)
        and isinstance(body.get("refresh_token"), str)
        and body.get("token_type") == "bearer"
        and isinstance(body.get("access_expires_in"), int)
        and isinstance(body.get("refresh_expires_in"), int)
        and isinstance(body.get("user"), dict)
    )


def user_shape(user):
    expected = {
        "user_id", "username", "role", "is_active",
        "created_at", "updated_at", "last_login_at",
    }
    return expected.issubset(user.keys())


def main():
    admin_password = load_bootstrap_password()
    user_password = f"Pw-{uuid.uuid4().hex}"       # 每次运行随机，不留固定口令
    user_password_new = f"Pw-{uuid.uuid4().hex}"

    # 1) 登录成功 + 结构（注意：登录会吊销同用户旧会话——与 Python 语义一致；
    #    跨端校验必须在"被验令牌的会话仍活跃"时进行，故按下方顺序交错执行）
    status, body = call(RS, "POST", "/auth/login", {"username": "admin", "password": admin_password})
    check("rust login ok", status == 200 and token_shape(body), f"{status} {body}")
    rust_bundle = body

    # 2) Python 接受 Rust 签发的令牌（此刻 Rust 会话活跃）
    status, body = call(PY, "GET", "/auth/me", token=rust_bundle["access_token"])
    check("python accepts rust token", status == 200 and body.get("username") == "admin", f"{status} {body}")

    # 3) Rust 再登录一次（吊销上面的会话），随后 Python 登录产生活跃会话
    status, body = call(RS, "POST", "/auth/login", {"username": "admin", "password": admin_password})
    check("rust re-login ok", status == 200 and token_shape(body), f"{status} {body}")
    status, body = call(PY, "POST", "/auth/login", {"username": "admin", "password": admin_password})
    check("python login ok", status == 200 and token_shape(body), f"{status} {body}")
    python_bundle = body

    # 4) Rust 接受 Python 签发的令牌（此刻 Python 会话活跃）
    status, body = call(RS, "GET", "/auth/me", token=python_bundle["access_token"])
    check("rust accepts py token", status == 200 and body.get("username") == "admin", f"{status} {body}")

    # 5) 密码错误
    for name, target in (("python", PY), ("rust", RS)):
        status, body = call(target, "POST", "/auth/login", {"username": "admin", "password": "definitely-wrong"})
        check(f"{name} login wrong pw 401", status == 401 and body.get("detail") == "用户名或密码错误", f"{status} {body}")

    # 6) user 字段集合一致
    check("user shape py", user_shape(python_bundle["user"]), str(python_bundle["user"].keys()))
    check("user shape rs", user_shape(rust_bundle["user"]), str(rust_bundle["user"].keys()))

    # 7) refresh 流程（在 Rust 端刷新 Python 令牌，旧会话吊销后 Python 端也必须拒绝）
    old_access = python_bundle["access_token"]
    status, body = call(RS, "POST", "/auth/refresh", {"refresh_token": python_bundle["refresh_token"]})
    check("rust refresh py token", status == 200 and token_shape(body), f"{status} {body}")
    status, _ = call(PY, "GET", "/auth/me", token=old_access)
    check("python rejects revoked session", status == 401, f"{status}")

    # 8) 用 refresh_token 冒充 access_token → 401
    status, body = call(PY, "POST", "/auth/login", {"username": "admin", "password": admin_password})
    py_bundle2 = body
    status, body = call(RS, "GET", "/auth/me", token=py_bundle2["refresh_token"])
    check("rust rejects refresh-as-access", status == 401, f"{status} {body}")

    # 7) 注册（Python 注册 → Rust 用同一账号登录 = 同库互通；口令每次随机）
    username = f"ct_{uuid.uuid4().hex[:10]}"
    status, body = call(PY, "POST", "/auth/register", {
        "username": username, "password": user_password, "confirm_password": user_password,
    })
    check("python register", status == 200 and token_shape(body), f"{status} {body}")
    status, body = call(PY, "POST", "/auth/register", {
        "username": username, "password": user_password, "confirm_password": user_password,
    })
    check("python register duplicate 409", status == 409 and body.get("detail") == "账号已存在", f"{status} {body}")
    status, body = call(RS, "POST", "/auth/login", {"username": username.upper(), "password": user_password})
    check("rust login user registered via python (case-insensitive)", status == 200 and body["user"]["username"] == username, f"{status} {body}")

    # 8) 改密（Rust 端改 → Python 端旧密码拒绝/新密码通过）
    new_access = body["access_token"]
    status, cbody = call(RS, "POST", "/auth/password", {
        "current_password": user_password, "new_password": user_password_new,
    }, token=new_access)
    check("rust change password", status == 200 and token_shape(cbody), f"{status} {cbody}")
    status, _ = call(PY, "POST", "/auth/login", {"username": username, "password": user_password})
    check("python rejects old password", status == 401, f"{status}")
    status, _ = call(PY, "POST", "/auth/login", {"username": username, "password": user_password_new})
    check("python accepts new password", status == 200, f"{status}")

    # 9) 未带 token 访问 /me
    for name, target in (("python", PY), ("rust", RS)):
        status, body = call(target, "GET", "/auth/me")
        check(f"{name} me without token 401", status == 401, f"{status} {body}")

    print()
    if failures:
        print(f"FAILED: {len(failures)} 项未通过: {failures}")
        raise SystemExit(1)
    print("ALL CONTRACT TESTS PASSED")


if __name__ == "__main__":
    main()
