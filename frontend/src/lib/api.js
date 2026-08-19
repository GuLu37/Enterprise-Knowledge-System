const defaultBaseUrl = '/api/v1'
const AUTH_TOKEN_KEY = 'enterprise-knowledge-system.auth-token'
const REFRESH_TOKEN_KEY = 'enterprise-knowledge-system.refresh-token'
const RENEWED_ACCESS_TOKEN_HEADER = 'X-Access-Token'
const JSON_HEADERS = {
  'Content-Type': 'application/json',
}

function getBaseUrl() {
  return (import.meta.env.VITE_API_BASE_URL || defaultBaseUrl).replace(/\/$/, '')
}

function buildUrl(path) {
  return `${getBaseUrl()}${path.startsWith('/') ? path : `/${path}`}`
}

function withJsonBody(options = {}, payload = {}) {
  return {
    ...options,
    headers: {
      ...JSON_HEADERS,
      ...(options.headers || {}),
    },
    body: JSON.stringify(payload),
  }
}

async function parseJsonResponse(response) {
  const text = await response.text()
  if (!text) return null
  try {
    return JSON.parse(text)
  } catch {
    return { detail: text }
  }
}

function formatErrorDetail(detail, fallback = '请求失败') {
  if (typeof detail === 'string' && detail.trim()) {
    return detail
  }

  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => {
        if (!item || typeof item !== 'object') return ''
        const location = Array.isArray(item.loc) ? item.loc.slice(1).join('.') : ''
        const message = typeof item.msg === 'string' ? item.msg : ''
        return [location, message].filter(Boolean).join('：')
      })
      .filter(Boolean)
    if (messages.length) return messages.join('；')
  }

  if (detail && typeof detail === 'object' && typeof detail.message === 'string') {
    return detail.message
  }

  return fallback
}

export function getAuthToken() {
  return localStorage.getItem(AUTH_TOKEN_KEY) || ''
}

export function getRefreshToken() {
  return localStorage.getItem(REFRESH_TOKEN_KEY) || ''
}

export function setAuthTokens({ accessToken = '', refreshToken = '' } = {}) {
  if (accessToken) {
    localStorage.setItem(AUTH_TOKEN_KEY, accessToken)
  } else {
    localStorage.removeItem(AUTH_TOKEN_KEY)
  }

  if (refreshToken) {
    localStorage.setItem(REFRESH_TOKEN_KEY, refreshToken)
  } else {
    localStorage.removeItem(REFRESH_TOKEN_KEY)
  }
}

export function setAuthToken(token) {
  if (token) {
    localStorage.setItem(AUTH_TOKEN_KEY, token)
  } else {
    localStorage.removeItem(AUTH_TOKEN_KEY)
  }
}

export function clearAuthToken() {
  localStorage.removeItem(AUTH_TOKEN_KEY)
  localStorage.removeItem(REFRESH_TOKEN_KEY)
}

function storeAuthTokens(result = {}) {
  setAuthTokens({
    accessToken: result.access_token,
    refreshToken: result.refresh_token,
  })
}

function buildHeaders(headers = {}, withAuth = true) {
  const nextHeaders = { ...headers }
  if (withAuth) {
    const token = getAuthToken()
    if (token) {
      nextHeaders.Authorization = `Bearer ${token}`
    }
  }
  return nextHeaders
}

function captureRenewedAccessToken(response) {
  const renewedAccessToken = response.headers.get(RENEWED_ACCESS_TOKEN_HEADER)
  if (renewedAccessToken) {
    setAuthToken(renewedAccessToken)
  }
}

async function refreshAuthSession() {
  const refreshToken = getRefreshToken()
  if (!refreshToken) {
    throw new Error('登录已失效，请重新登录。')
  }

  const result = await requestJson('/auth/refresh', {
    ...withJsonBody({ method: 'POST', auth: false }, { refresh_token: refreshToken }),
  }, false)

  storeAuthTokens(result)
  return result
}

async function requestJson(path, options = {}, allowRetry = true) {
  const { auth = true, headers = {}, ...fetchOptions } = options
  const response = await fetch(buildUrl(path), {
    ...fetchOptions,
    headers: buildHeaders(headers, auth),
  })
  captureRenewedAccessToken(response)

  if (!response.ok) {
    // 先尝试用刷新令牌恢复会话，再决定是否把错误抛给上层。
    if (response.status === 401 && auth && allowRetry && getRefreshToken()) {
      try {
        await refreshAuthSession()
        return requestJson(path, options, false)
      } catch {
        clearAuthToken()
      }
    }
    const detail = formatErrorDetail((await parseJsonResponse(response))?.detail)
    throw new Error(detail)
  }
  return response.json()
}

export async function listDocuments(params = {}) {
  const search = new URLSearchParams()
  if (params.skip != null) search.set('skip', String(params.skip))
  if (params.limit != null) search.set('limit', String(params.limit))

  return requestJson(`/documents/list?${search.toString()}`)
}

export async function uploadDocument(file) {
  const formData = new FormData()
  formData.append('file', file)

  return requestJson('/documents/upload', {
    method: 'POST',
    body: formData,
  })
}

export async function deleteDocument(documentId) {
  return requestJson(`/documents/delete/${documentId}`, {
    method: 'DELETE',
  })
}

export async function getDocumentContent(documentId) {
  return requestJson(`/documents/content/${documentId}`)
}

export async function getChatSettings() {
  return requestJson('/chat/settings')
}

export async function getChatRuntimeStatus() {
  return requestJson('/chat/runtime-status')
}

export async function warmupChatRuntime() {
  return requestJson('/chat/warmup', {
    method: 'POST',
  })
}

export async function deleteConversation(conversationId) {
  return requestJson(`/chat/conversations/${conversationId}`, {
    method: 'DELETE',
  })
}

export async function generateChat(payload) {
  return requestJson('/chat/generate', withJsonBody({ method: 'POST' }, payload))
}

export async function login(payload) {
  const result = await requestJson('/auth/login', {
    ...withJsonBody({ method: 'POST', auth: false }, payload),
  })
  storeAuthTokens(result)
  return result
}

export async function register(payload) {
  const result = await requestJson('/auth/register', {
    ...withJsonBody({ method: 'POST', auth: false }, payload),
  })
  storeAuthTokens(result)
  return result
}

export async function getCurrentUser() {
  return requestJson('/auth/me')
}

export async function refreshLogin() {
  return refreshAuthSession()
}

export async function changePassword(payload) {
  const result = await requestJson('/auth/password', {
    ...withJsonBody({ method: 'POST' }, payload),
  })
  storeAuthTokens(result)
  return result
}

function parseSseEventBlock(block) {
  // SSE 的单个事件块可能包含多行 data，这里先按行拆分再恢复成一条事件。
  const lines = block.split('\n')
  let event = 'message'
  const dataLines = []

  for (const line of lines) {
    if (line.startsWith('event:')) {
      event = line.slice(6).trim()
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).trim())
    }
  }

  if (!dataLines.length) return null

  const raw = dataLines.join('\n')
  try {
    return { event, data: JSON.parse(raw) }
  } catch {
    return { event, data: raw }
  }
}

export async function streamChat(payload, onEvent) {
  // 先发起一次流式请求，若鉴权过期则补刷一次刷新令牌后重试。
  const requestInit = {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(getAuthToken() ? { Authorization: `Bearer ${getAuthToken()}` } : {}),
    },
    body: JSON.stringify(payload),
  }

  let response = await fetch(buildUrl('/chat/stream'), requestInit)
  captureRenewedAccessToken(response)
  if (response.status === 401 && getRefreshToken()) {
    try {
      await refreshAuthSession()
      response = await fetch(buildUrl('/chat/stream'), {
        ...requestInit,
        headers: {
          ...requestInit.headers,
          ...(getAuthToken() ? { Authorization: `Bearer ${getAuthToken()}` } : {}),
        },
      })
      captureRenewedAccessToken(response)
    } catch {
      clearAuthToken()
    }
  }

  if (!response.ok || !response.body) {
    throw new Error(formatErrorDetail(
      (await parseJsonResponse(response))?.detail,
      '流式请求失败',
    ))
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''
  let sawDone = false

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n')

    let boundaryIndex = buffer.indexOf('\n\n')
    while (boundaryIndex !== -1) {
      const block = buffer.slice(0, boundaryIndex).trim()
      buffer = buffer.slice(boundaryIndex + 2)
      if (block) {
        const parsed = parseSseEventBlock(block)
        if (parsed) {
          if (parsed.event === 'done') sawDone = true
          onEvent(parsed)
        }
      }
      boundaryIndex = buffer.indexOf('\n\n')
    }
  }

  const tail = buffer.trim()
  if (tail) {
    const parsed = parseSseEventBlock(tail)
    if (parsed) {
      if (parsed.event === 'done') sawDone = true
      onEvent(parsed)
    }
  }

  if (!sawDone) {
    onEvent({ event: 'done', data: {} })
  }
}
