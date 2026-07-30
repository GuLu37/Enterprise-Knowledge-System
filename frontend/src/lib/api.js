const defaultBaseUrl = '/api/v1'
const AUTH_TOKEN_KEY = 'enterprise-knowledge-system.auth-token'
const REFRESH_TOKEN_KEY = 'enterprise-knowledge-system.refresh-token'

function getBaseUrl() {
  return (import.meta.env.VITE_API_BASE_URL || defaultBaseUrl).replace(/\/$/, '')
}

function buildUrl(path) {
  return `${getBaseUrl()}${path.startsWith('/') ? path : `/${path}`}`
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
  setAuthTokens({ accessToken: token })
}

export function clearAuthToken() {
  localStorage.removeItem(AUTH_TOKEN_KEY)
  localStorage.removeItem(REFRESH_TOKEN_KEY)
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

async function refreshAuthSession() {
  const refreshToken = getRefreshToken()
  if (!refreshToken) {
    throw new Error('登录已失效，请重新登录。')
  }

  const result = await requestJson('/auth/refresh', {
    method: 'POST',
    auth: false,
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ refresh_token: refreshToken }),
  }, false)

  setAuthTokens({
    accessToken: result.access_token,
    refreshToken: result.refresh_token,
  })
  return result
}

async function requestJson(path, options = {}, allowRetry = true) {
  const { auth = true, headers = {}, ...fetchOptions } = options
  const response = await fetch(buildUrl(path), {
    ...fetchOptions,
    headers: buildHeaders(headers, auth),
  })

  if (!response.ok) {
    const detail = (await parseJsonResponse(response))?.detail || '请求失败'
    if (response.status === 401 && auth && allowRetry && getRefreshToken()) {
      try {
        await refreshAuthSession()
        return requestJson(path, options, false)
      } catch {
        clearAuthToken()
      }
    }
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
  return requestJson('/chat/generate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })
}

export async function login(payload) {
  const result = await requestJson('/auth/login', {
    method: 'POST',
    auth: false,
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })
  setAuthTokens({
    accessToken: result.access_token,
    refreshToken: result.refresh_token,
  })
  return result
}

export async function register(payload) {
  const result = await requestJson('/auth/register', {
    method: 'POST',
    auth: false,
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })
  setAuthTokens({
    accessToken: result.access_token,
    refreshToken: result.refresh_token,
  })
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
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })
  setAuthTokens({
    accessToken: result.access_token,
    refreshToken: result.refresh_token,
  })
  return result
}

function parseSseEventBlock(block) {
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
  const requestInit = {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(getAuthToken() ? { Authorization: `Bearer ${getAuthToken()}` } : {}),
    },
    body: JSON.stringify(payload),
  }

  let response = await fetch(buildUrl('/chat/stream'), requestInit)
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
    } catch {
      clearAuthToken()
    }
  }

  if (!response.ok || !response.body) {
    throw new Error((await parseJsonResponse(response))?.detail || '流式请求失败')
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
