<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import {
  ArrowRight,
  Bot,
  Check,
  ChevronDown,
  CircleAlert,
  FileText,
  Loader2,
  LogIn,
  LogOut,
  KeyRound,
  MessageSquare,
  Paperclip,
  Plus,
  RefreshCw,
  Send,
  Trash2,
  Upload,
  User,
  Wand2,
  X,
} from 'lucide-vue-next'
import {
  deleteConversation,
  deleteDocument,
  changePassword as apiChangePassword,
  generateChat,
  getChatSettings,
  getCurrentUser,
  getDocumentContent,
  listDocuments,
  login as apiLogin,
  register as apiRegister,
  streamChat,
  uploadDocument,
  clearAuthToken,
  getAuthToken,
} from './lib/api'

const CONVERSATION_STORAGE_KEY = 'enterprise-knowledge-system.conversations'
const DEFAULT_MAX_CONVERSATIONS = 20

function createAssistantGreeting(content = '你好，我在这里。可以先上传文档，再直接提问。') {
  return {
    id: crypto.randomUUID(),
    role: 'assistant',
    content,
    sources: [],
    status: 'idle',
    ephemeral: true,
  }
}

function createConversation() {
  const now = new Date().toISOString()
  return {
    conversation_id: crypto.randomUUID(),
    session_id: crypto.randomUUID(),
    created_at: now,
    updated_at: now,
    messages: [createAssistantGreeting()],
  }
}

function normalizeStoredConversation(item) {
  const fallback = createConversation()
  const messages = Array.isArray(item?.messages) && item.messages.length
    ? item.messages
    : fallback.messages

  return {
    conversation_id: item?.conversation_id || fallback.conversation_id,
    session_id: item?.session_id || fallback.session_id,
    created_at: item?.created_at || fallback.created_at,
    updated_at: item?.updated_at || item?.created_at || fallback.updated_at,
    messages,
  }
}

function loadConversations() {
  try {
    const stored = JSON.parse(localStorage.getItem(CONVERSATION_STORAGE_KEY) || '[]')
    if (Array.isArray(stored) && stored.length) {
      return stored.map(normalizeStoredConversation)
    }
  } catch {
    // Ignore corrupt browser storage and start a clean local conversation list.
  }
  return [createConversation()]
}

const isAuthenticated = ref(false)
const authLoading = ref(true)
const authSubmitting = ref(false)
const authError = ref('')
const currentUser = ref(null)
const authMode = ref('login')
const loginForm = ref({
  username: '',
  password: '',
  confirmPassword: '',
})
const passwordModalOpen = ref(false)
const passwordSubmitting = ref(false)
const passwordError = ref('')
const passwordForm = ref({
  currentPassword: '',
  newPassword: '',
  confirmPassword: '',
})
const profileMenuOpen = ref(false)

const docs = ref([])
const docsLoading = ref(false)
const selectedDocId = ref('')
const uploadError = ref('')
const uploadBusy = ref(false)
const refreshingDocs = ref(false)
const conversations = ref(loadConversations())
const activeConversationId = ref(conversations.value[0].conversation_id)
const chatMessages = ref(conversations.value[0].messages)
const chatSettings = ref({ max_conversations: DEFAULT_MAX_CONVERSATIONS })
const deletingConversationIds = ref({})
const streamingConversationIds = ref({})
const inputText = ref('')
const chatBusy = computed(() => Boolean(streamingConversationIds.value[activeConversationId.value]))
const streamMode = ref(true)
const useRetrieval = ref(true)
const retrievalMethod = ref('hybrid')
const retrievalMenuOpen = ref(false)
const errorText = ref('')
const conversationId = ref(conversations.value[0].conversation_id)
const sessionId = ref(conversations.value[0].session_id)
const selectedDoc = computed(() => docs.value.find((item) => item.document_id === selectedDocId.value) || null)
const selectedDocContent = ref('')
const selectedDocLoading = ref(false)
const selectedDocError = ref('')
const scrollHost = ref(null)
const fileInput = ref(null)
const retrievalMenuRef = ref(null)
const autoScrollEnabled = ref(true)
const expandedSources = ref({})
let previewRequestSeq = 0
let sheetDragCleanup = null
let conversationSaveTimer = null
let profileMenuCloseTimer = null

const retrievalOptions = [
  { value: 'hybrid', label: '混合检索' },
  { value: 'dense', label: '密集检索' },
  { value: 'sparse', label: '稀疏检索' },
]

const currentRetrievalLabel = computed(
  () => retrievalOptions.find((option) => option.value === retrievalMethod.value)?.label || '混合检索',
)

const maxConversations = computed(() => {
  const configured = Number(chatSettings.value.max_conversations)
  return Number.isFinite(configured) && configured > 0 ? configured : DEFAULT_MAX_CONVERSATIONS
})

const conversationLimitReached = computed(() => conversations.value.length >= maxConversations.value)

const conversationList = computed(() => (
  [...conversations.value].sort((left, right) => (
    new Date(right.updated_at).getTime() - new Date(left.updated_at).getTime()
  ))
))

function getConversationTitle(conversation) {
  const firstQuestion = conversation.messages?.find((message) => message.role === 'user' && message.content?.trim())
  return trimText(firstQuestion?.content || '新对话', 34)
}

function getConversationPreview(conversation) {
  const latest = [...(conversation.messages || [])]
    .reverse()
    .find((message) => !message.ephemeral && message.content?.trim())
  return trimText(latest?.content || '还没有开始提问', 46)
}

function getConversationTurnCount(conversation) {
  return (conversation.messages || []).filter((message) => message.role === 'user' && !message.ephemeral).length
}

function findConversation(id = activeConversationId.value) {
  return conversations.value.find((conversation) => conversation.conversation_id === id)
}

function isConversationStreaming(id) {
  return Boolean(streamingConversationIds.value[id])
}

function setConversationStreaming(id, isStreaming) {
  if (!id) return
  const next = { ...streamingConversationIds.value }
  if (isStreaming) {
    next[id] = true
  } else {
    delete next[id]
  }
  streamingConversationIds.value = next
}

function touchConversation(id, shouldUpdateTime = true) {
  const conversation = findConversation(id)
  if (!conversation) return
  if (shouldUpdateTime) {
    conversation.updated_at = new Date().toISOString()
  }
  if (activeConversationId.value === id) {
    chatMessages.value = conversation.messages
    conversationId.value = conversation.conversation_id
    sessionId.value = conversation.session_id
  }
}

function touchActiveConversation(shouldUpdateTime = true) {
  const active = findConversation()
  if (!active) return
  active.messages = chatMessages.value
  active.conversation_id = conversationId.value
  active.session_id = sessionId.value
  if (shouldUpdateTime) {
    active.updated_at = new Date().toISOString()
  }
}

function activateConversation(id) {
  touchActiveConversation(false)
  const target = findConversation(id)
  if (!target) return

  activeConversationId.value = target.conversation_id
  conversationId.value = target.conversation_id
  sessionId.value = target.session_id
  chatMessages.value = target.messages
  inputText.value = ''
  errorText.value = ''
  expandedSources.value = {}
  autoScrollEnabled.value = true
  nextTick(() => scrollToBottom(true))
}

async function loadChatSettings() {
  try {
    const settings = await getChatSettings()
    chatSettings.value = {
      max_conversations: settings.max_conversations || DEFAULT_MAX_CONVERSATIONS,
    }
  } catch (error) {
    errorText.value = error.message || '聊天配置加载失败，已使用默认对话上限。'
  }
}

async function bootstrapAuth() {
  authLoading.value = true
  authError.value = ''

  const token = getAuthToken()
  if (!token) {
    isAuthenticated.value = false
    authLoading.value = false
    return false
  }

  try {
    currentUser.value = await getCurrentUser()
    isAuthenticated.value = true
    return true
  } catch (error) {
    clearAuthToken()
    currentUser.value = null
    isAuthenticated.value = false
    authError.value = error.message || '登录态已失效，请重新登录。'
    return false
  } finally {
    authLoading.value = false
  }
}

function switchAuthMode(mode) {
  if (authMode.value === mode) return
  authMode.value = mode
  authError.value = ''
  loginForm.value.password = ''
  loginForm.value.confirmPassword = ''
}

async function handleLogin() {
  if (authSubmitting.value) return
  authSubmitting.value = true
  authError.value = ''

  try {
    const payload = {
      username: loginForm.value.username.trim(),
      password: loginForm.value.password,
    }

    if (authMode.value === 'register') {
      if (loginForm.value.password !== loginForm.value.confirmPassword) {
        authError.value = '两次输入的密码不一致'
        return
      }
      await apiRegister({
        ...payload,
        confirm_password: loginForm.value.confirmPassword,
      })
    } else {
      await apiLogin(payload)
    }
    window.location.reload()
  } catch (error) {
    authError.value = error.message || '登录失败'
  } finally {
    authSubmitting.value = false
  }
}

function handleLogout() {
  closeProfileMenu()
  clearAuthToken()
  window.location.reload()
}

function openProfileMenu() {
  if (profileMenuCloseTimer) {
    clearTimeout(profileMenuCloseTimer)
    profileMenuCloseTimer = null
  }
  profileMenuOpen.value = true
}

function closeProfileMenu() {
  if (profileMenuCloseTimer) {
    clearTimeout(profileMenuCloseTimer)
    profileMenuCloseTimer = null
  }
  profileMenuOpen.value = false
}

function scheduleCloseProfileMenu() {
  if (profileMenuCloseTimer) {
    clearTimeout(profileMenuCloseTimer)
  }
  profileMenuCloseTimer = window.setTimeout(() => {
    profileMenuOpen.value = false
    profileMenuCloseTimer = null
  }, 180)
}

function openPasswordModal() {
  closeProfileMenu()
  passwordError.value = ''
  passwordForm.value = {
    currentPassword: '',
    newPassword: '',
    confirmPassword: '',
  }
  passwordModalOpen.value = true
}

function closePasswordModal() {
  passwordModalOpen.value = false
  passwordError.value = ''
  passwordForm.value = {
    currentPassword: '',
    newPassword: '',
    confirmPassword: '',
  }
}

async function handleChangePassword() {
  if (passwordSubmitting.value) return
  passwordError.value = ''

  if (passwordForm.value.newPassword !== passwordForm.value.confirmPassword) {
    passwordError.value = '两次输入的新密码不一致'
    return
  }

  passwordSubmitting.value = true
  try {
    await apiChangePassword({
      current_password: passwordForm.value.currentPassword,
      new_password: passwordForm.value.newPassword,
    })
    closeProfileMenu()
    closePasswordModal()
    window.location.reload()
  } catch (error) {
    passwordError.value = error.message || '修改密码失败'
  } finally {
    passwordSubmitting.value = false
  }
}

async function removeConversation(conversation) {
  if (!conversation?.conversation_id) return
  const id = conversation.conversation_id
  if (isConversationStreaming(id)) {
    errorText.value = '这个对话还在输出中，完成后再删除。'
    return
  }

  if (!window.confirm(`删除对话「${getConversationTitle(conversation)}」并同步清除长期记忆？`)) return

  deletingConversationIds.value = {
    ...deletingConversationIds.value,
    [id]: true,
  }

  try {
    await deleteConversation(id)
    conversations.value = conversations.value.filter((item) => item.conversation_id !== id)

    if (!conversations.value.length) {
      const nextConversation = createConversation()
      conversations.value.push(nextConversation)
    }

    if (activeConversationId.value === id) {
      const nextActive = conversationList.value[0] || conversations.value[0]
      activeConversationId.value = nextActive.conversation_id
      conversationId.value = nextActive.conversation_id
      sessionId.value = nextActive.session_id
      chatMessages.value = nextActive.messages
      expandedSources.value = {}
      autoScrollEnabled.value = true
      await nextTick()
      scrollToBottom(true)
    }
  } catch (error) {
    errorText.value = error.message || '删除对话失败'
  } finally {
    const nextDeleting = { ...deletingConversationIds.value }
    delete nextDeleting[id]
    deletingConversationIds.value = nextDeleting
  }
}

function persistConversations() {
  localStorage.setItem(CONVERSATION_STORAGE_KEY, JSON.stringify(conversations.value))
}

function scheduleConversationPersist() {
  if (conversationSaveTimer) {
    clearTimeout(conversationSaveTimer)
  }
  conversationSaveTimer = window.setTimeout(() => {
    conversationSaveTimer = null
    persistConversations()
  }, 250)
}

function trimText(text, limit = 90) {
  const normalized = (text || '').replace(/\s+/g, ' ').trim()
  if (!normalized) return ''
  return normalized.length > limit ? `${normalized.slice(0, limit)}...` : normalized
}

function toggleRetrievalMenu() {
  retrievalMenuOpen.value = !retrievalMenuOpen.value
}

function chooseRetrievalMethod(value) {
  retrievalMethod.value = value
  retrievalMenuOpen.value = false
}

function handleRetrievalMenuClick(event) {
  if (retrievalMenuRef.value && !retrievalMenuRef.value.contains(event.target)) {
    retrievalMenuOpen.value = false
  }
}

function toggleSources(messageId) {
  expandedSources.value = {
    ...expandedSources.value,
    [messageId]: !expandedSources.value[messageId],
  }
}

function updateConversationMessage(targetConversationId, messageId, updater, shouldUpdateTime = false) {
  const conversation = findConversation(targetConversationId)
  const target = conversation?.messages.find((message) => message.id === messageId)
  if (target) {
    updater(target)
    touchConversation(targetConversationId, shouldUpdateTime)
  }
}

function updateChatMessage(messageId, updater, shouldUpdateTime = false) {
  updateConversationMessage(activeConversationId.value, messageId, updater, shouldUpdateTime)
}

function readStreamContent(eventData) {
  if (!eventData) return ''
  if (typeof eventData === 'string') return eventData
  return eventData.content || eventData.delta || eventData.response || eventData.text || ''
}

function extractHighlights(text, limit = 4) {
  const normalized = (text || '').trim()
  if (!normalized) return []

  const lines = normalized
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)

  const bulletLines = lines
    .filter((line) => /^([\-*•]|\d+[.)])\s+/.test(line))
    .map((line) => line.replace(/^([\-*•]|\d+[.)])\s+/, '').trim())

  if (bulletLines.length) {
    return bulletLines.slice(0, limit).map((item) => trimText(item, 72))
  }

  return normalized
    .replace(/\s+/g, ' ')
    .split(/[。！？!?；;]+/)
    .map((item) => item.trim())
    .filter(Boolean)
    .slice(0, limit)
    .map((item) => trimText(item, 72))
}

function parseCsvLine(line) {
  const cells = []
  let current = ''
  let inQuotes = false

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i]
    const next = line[i + 1]

    if (char === '"') {
      if (inQuotes && next === '"') {
        current += '"'
        i += 1
      } else {
        inQuotes = !inQuotes
      }
      continue
    }

    if (char === ',' && !inQuotes) {
      cells.push(current)
      current = ''
      continue
    }

    current += char
  }

  cells.push(current)
  return cells.map((cell) => cell.trim())
}

function parseSpreadsheetPreview(content) {
  const lines = (content || '').replace(/\r\n/g, '\n').split('\n')
  const sheets = []
  let currentSheet = null

  const commitSheet = () => {
    if (!currentSheet) return
    const rows = currentSheet.rows.filter((row) => row.some((cell) => cell !== ''))
    if (rows.length) {
      const headers = rows[0]
      const dataRows = rows.slice(1)
      const columnCount = Math.max(
        headers.length,
        ...dataRows.map((row) => row.length),
      )
      currentSheet.headers = Array.from({ length: columnCount }, (_, index) => headers[index] || '')
      currentSheet.rows = dataRows.map((row) => Array.from({ length: columnCount }, (_, index) => row[index] || ''))
      sheets.push(currentSheet)
    }
  }

  for (const rawLine of lines) {
    const line = rawLine.trimEnd()
    if (!line.trim()) continue

    if (line.startsWith('[Sheet] ')) {
      commitSheet()
      currentSheet = {
        name: line.replace('[Sheet] ', '').trim() || 'Sheet',
        headers: [],
        rows: [],
      }
      continue
    }

    if (!currentSheet) {
      currentSheet = {
        name: 'Sheet',
        headers: [],
        rows: [],
      }
    }

    currentSheet.rows.push(parseCsvLine(line))
  }

  commitSheet()
  return sheets
}

const spreadsheetPreview = computed(() => {
  if (!selectedDoc.value) return []
  if (!['xlsx', 'xls'].includes((selectedDoc.value.file_type || '').toLowerCase())) return []
  return parseSpreadsheetPreview(selectedDocContent.value)
})

function stopSheetDrag() {
  if (typeof sheetDragCleanup === 'function') {
    sheetDragCleanup()
    sheetDragCleanup = null
  }
}

function startSheetDrag(event) {
  if (event.button !== 0) return
  const wrap = event.currentTarget
  if (!wrap || !wrap.scrollWidth || wrap.scrollWidth <= wrap.clientWidth) return

  const state = {
    startX: event.clientX,
    startScrollLeft: wrap.scrollLeft,
    pointerId: event.pointerId,
    element: wrap,
  }

  wrap.classList.add('is-dragging')
  wrap.setPointerCapture?.(event.pointerId)

  const onMove = (moveEvent) => {
    if (moveEvent.pointerId !== state.pointerId) return
    moveEvent.preventDefault()
    const delta = moveEvent.clientX - state.startX
    wrap.scrollLeft = state.startScrollLeft - delta
  }

  const finishDrag = (upEvent) => {
    if (upEvent && upEvent.pointerId !== state.pointerId) return
    wrap.classList.remove('is-dragging')
    wrap.releasePointerCapture?.(state.pointerId)
    window.removeEventListener('pointermove', onMove)
    window.removeEventListener('pointerup', finishDrag)
    window.removeEventListener('pointercancel', finishDrag)
    sheetDragCleanup = null
  }

  window.addEventListener('pointermove', onMove)
  window.addEventListener('pointerup', finishDrag)
  window.addEventListener('pointercancel', finishDrag)
  sheetDragCleanup = finishDrag
}

const conversationSummary = computed(() => {
  const visibleMessages = chatMessages.value.filter((message) => !message.ephemeral && message.content?.trim())
  const userMessages = visibleMessages.filter((message) => message.role === 'user')
  const assistantMessages = visibleMessages.filter((message) => message.role === 'assistant')

  const recentQuestions = userMessages.slice(-4).map((message) => trimText(message.content, 80))
  const keyPoints = assistantMessages
    .slice(-4)
    .flatMap((message) => extractHighlights(message.content, 2))
    .slice(0, 6)

  return {
    turnCount: userMessages.length,
    currentFocus: recentQuestions.at(-1) || trimText(visibleMessages.at(-1)?.content || '', 80),
    recentQuestions,
    keyPoints,
  }
})

function formatBytes(size = 0) {
  if (!size) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  let value = size
  let unitIndex = 0
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024
    unitIndex += 1
  }
  return `${value.toFixed(value >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`
}

function formatDate(value) {
  if (!value) return '-'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? '-' : date.toLocaleString('zh-CN')
}

function statusLabel(status) {
  return {
    processing: '处理中',
    ready: '可用',
    failed: '失败',
    deleting: '删除中',
    deleted: '已删除',
  }[status] || status
}

function statusClass(status) {
  return {
    processing: 'is-processing',
    ready: 'is-ready',
    failed: 'is-failed',
    deleting: 'is-processing',
    deleted: 'is-muted',
  }[status] || ''
}

function buildHistoryPayload(messages = chatMessages.value) {
  return messages
    .slice(0, -2)
    .filter((message) => (message.role === 'user' || message.role === 'assistant') && !message.ephemeral)
    .map((message) => ({
      role: message.role,
      content: message.content,
    }))
}

async function loadDocs() {
  docsLoading.value = true
  errorText.value = ''
  try {
    const result = await listDocuments({ skip: 0, limit: 100 })
    docs.value = result.documents || []
    if (selectedDocId.value && !docs.value.some((doc) => doc.document_id === selectedDocId.value)) {
      selectedDocId.value = ''
    }
  } catch (error) {
    errorText.value = error.message || '加载文档失败'
  } finally {
    docsLoading.value = false
  }
}

async function refreshDocs() {
  refreshingDocs.value = true
  try {
    await loadDocs()
  } finally {
    refreshingDocs.value = false
  }
}

async function loadSelectedDocumentContent(documentId) {
  if (!documentId) {
    selectedDocContent.value = ''
    selectedDocError.value = ''
    selectedDocLoading.value = false
    return
  }

  const requestSeq = ++previewRequestSeq
  selectedDocLoading.value = true
  selectedDocError.value = ''
  selectedDocContent.value = ''
  try {
    const result = await getDocumentContent(documentId)
    if (requestSeq === previewRequestSeq) {
      selectedDocContent.value = result.content || ''
    }
  } catch (error) {
    if (requestSeq === previewRequestSeq) {
      selectedDocError.value = error.message || '获取文档内容失败'
    }
  } finally {
    if (requestSeq === previewRequestSeq) {
      selectedDocLoading.value = false
    }
  }
}

function selectDocument(doc) {
  if (!doc?.document_id) return
  if (selectedDocId.value === doc.document_id) {
    selectedDocId.value = ''
    selectedDocContent.value = ''
    selectedDocError.value = ''
    selectedDocLoading.value = false
    previewRequestSeq += 1
    return
  }
  selectedDocId.value = doc.document_id
}

function closeDocumentPreview() {
  selectedDocId.value = ''
  selectedDocContent.value = ''
  selectedDocError.value = ''
  selectedDocLoading.value = false
  previewRequestSeq += 1
}

function openPicker() {
  fileInput.value?.click()
}

async function handleUpload(event) {
  const file = event.target.files?.[0]
  if (!file) return
  uploadBusy.value = true
  uploadError.value = ''
  try {
    const result = await uploadDocument(file)
    selectedDocId.value = result.document_id
    await loadDocs()
  } catch (error) {
    uploadError.value = error.message || '上传失败'
  } finally {
    uploadBusy.value = false
    event.target.value = ''
  }
}

async function handleDelete(doc) {
  if (!doc?.document_id) return
  if (doc.status !== 'ready') {
    errorText.value = '文档还在处理中，状态切换为可用后才可以删除。'
    return
  }
  if (!window.confirm(`删除文档「${doc.original_filename}」？`)) return
  try {
    await deleteDocument(doc.document_id)
    if (selectedDocId.value === doc.document_id) {
      closeDocumentPreview()
    }
    await loadDocs()
  } catch (error) {
    errorText.value = error.message || '删除失败'
  }
}

async function sendMessage() {
  const text = inputText.value.trim()
  if (!text || chatBusy.value) return

  errorText.value = ''
  const runConversationId = activeConversationId.value
  const conversation = findConversation(runConversationId)
  if (!conversation) return

  const userMessage = {
    id: crypto.randomUUID(),
    role: 'user',
    content: text,
    sources: [],
    status: 'done',
  }
  conversation.messages.push(userMessage)
  inputText.value = ''

  const assistantMessage = {
    id: crypto.randomUUID(),
    role: 'assistant',
    content: '',
    sources: [],
    status: 'streaming',
  }
  conversation.messages.push(assistantMessage)
  touchConversation(runConversationId)
  setConversationStreaming(runConversationId, true)
  await nextTick()
  scrollToBottom(true)

  const payload = {
    query: text,
    history: buildHistoryPayload(conversation.messages),
    conversation_id: conversation.conversation_id,
    session_id: conversation.session_id,
    use_retrieval: useRetrieval.value,
    stream: streamMode.value,
    retrieval_method: retrievalMethod.value,
    short_memory_strategy: 'window',
    short_memory_n: 5,
    short_memory_m: 10,
    top_k: 5,
  }

  try {
    if (payload.stream) {
      updateConversationMessage(runConversationId, assistantMessage.id, (message) => {
        message.status = 'streaming'
      })
      await streamChat(payload, (event) => {
        if (event.event === 'message') {
          const content = readStreamContent(event.data)
          if (content) {
            updateConversationMessage(runConversationId, assistantMessage.id, (message) => {
              message.content += content
            })
          }
          if (activeConversationId.value === runConversationId) {
            scrollToBottom()
          }
          return
        }
        if (event.event === 'tool_call') {
          updateConversationMessage(runConversationId, assistantMessage.id, (message) => {
            message.status = 'thinking'
          })
          return
        }
        if (event.event === 'tool_result') {
          updateConversationMessage(runConversationId, assistantMessage.id, (message) => {
            message.sources = event.data?.sources || message.sources
          })
          return
        }
        if (event.event === 'done') {
          updateConversationMessage(runConversationId, assistantMessage.id, (message) => {
            message.sources = event.data?.sources || message.sources
            message.status = 'done'
          }, true)
          return
        }
        if (event.event === 'error') {
          throw new Error(event.data?.message || '流式输出失败')
        }
      })
    } else {
      const result = await generateChat(payload)
      updateConversationMessage(runConversationId, assistantMessage.id, (message) => {
        message.content = result.response || ''
        message.sources = result.sources || []
        message.status = 'done'
      }, true)
    }
  } catch (error) {
    updateConversationMessage(runConversationId, assistantMessage.id, (message) => {
      message.content = error.message || '请求失败'
      message.status = 'error'
    }, true)
  } finally {
    setConversationStreaming(runConversationId, false)
    touchConversation(runConversationId)
    await nextTick()
  }
}

function isNearBottom(el) {
  const threshold = 120
  return el.scrollHeight - el.scrollTop - el.clientHeight <= threshold
}

function handleChatScroll() {
  const el = scrollHost.value
  if (!el) return
  autoScrollEnabled.value = isNearBottom(el)
}

function scrollToBottom(force = false) {
  const el = scrollHost.value
  if (!el) return
  if (!force && !autoScrollEnabled.value) return
  el.scrollTop = el.scrollHeight
}

function resetChat() {
  if (conversationLimitReached.value) {
    errorText.value = `已达到后端配置的对话上限 ${maxConversations.value}，请先删除不需要的对话。`
    return
  }

  touchActiveConversation(false)
  const nextConversation = createConversation()
  nextConversation.messages = [createAssistantGreeting('已开启新对话，可以继续提问。')]
  conversations.value.push(nextConversation)
  activeConversationId.value = nextConversation.conversation_id
  conversationId.value = nextConversation.conversation_id
  sessionId.value = nextConversation.session_id
  chatMessages.value = nextConversation.messages
  inputText.value = ''
  errorText.value = ''
  expandedSources.value = {}
  autoScrollEnabled.value = true
  nextTick(() => scrollToBottom(true))
}

function handleComposerKeydown(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    sendMessage()
  }
}

onMounted(async () => {
  const authenticated = await bootstrapAuth()
  if (!authenticated) return

  await Promise.all([
    loadDocs(),
    loadChatSettings(),
  ])
  await nextTick()
  scrollToBottom(true)
  document.addEventListener('click', handleRetrievalMenuClick)
})

onBeforeUnmount(() => {
  stopSheetDrag()
  if (conversationSaveTimer) {
    clearTimeout(conversationSaveTimer)
    conversationSaveTimer = null
  }
  if (profileMenuCloseTimer) {
    clearTimeout(profileMenuCloseTimer)
    profileMenuCloseTimer = null
  }
  persistConversations()
  document.removeEventListener('click', handleRetrievalMenuClick)
})

watch(selectedDocId, async (documentId) => {
  await loadSelectedDocumentContent(documentId)
}, { immediate: true })

watch(conversations, () => {
  scheduleConversationPersist()
}, { deep: true })
</script>

<template>
  <div v-if="authLoading" class="auth-shell">
    <div class="auth-panel">
      <Loader2 :size="22" class="spin" />
      <div>
        <div class="eyebrow">Enterprise Knowledge System</div>
        <div class="auth-panel__title">正在校验登录状态...</div>
      </div>
    </div>
  </div>

  <div v-else-if="!isAuthenticated" class="auth-shell">
    <form class="auth-panel auth-panel--form" @submit.prevent="handleLogin">
      <div class="auth-panel__header">
        <div class="auth-panel__icon">
          <component :is="authMode === 'login' ? LogIn : User" :size="18" />
        </div>
        <div>
          <div class="eyebrow">Enterprise Knowledge System</div>
          <h1>{{ authMode === 'login' ? '系统登录' : '创建账号' }}</h1>
        </div>
      </div>

      <div class="auth-tabs">
        <button
          class="auth-tab"
          type="button"
          :class="{ active: authMode === 'login' }"
          @click="switchAuthMode('login')"
        >
          登录
        </button>
        <button
          class="auth-tab"
          type="button"
          :class="{ active: authMode === 'register' }"
          @click="switchAuthMode('register')"
        >
          注册
        </button>
      </div>

      <label class="field">
        <span>用户名</span>
        <div class="field__input">
          <User :size="16" />
          <input
            v-model="loginForm.username"
            type="text"
            autocomplete="username"
            placeholder="请输入用户名"
          />
        </div>
      </label>

      <label class="field">
        <span>密码</span>
        <div class="field__input">
          <input
            v-model="loginForm.password"
            type="password"
            autocomplete="current-password"
            placeholder="请输入密码"
          />
        </div>
      </label>

      <label v-if="authMode === 'register'" class="field">
        <span>重复密码</span>
        <div class="field__input">
          <input
            v-model="loginForm.confirmPassword"
            type="password"
            autocomplete="new-password"
            placeholder="再次输入密码"
          />
        </div>
      </label>

      <p v-if="authError" class="inline-error auth-error">
        <CircleAlert :size="14" />
        <span>{{ authError }}</span>
      </p>

      <button class="primary-button auth-button" type="submit" :disabled="authSubmitting">
        <Loader2 v-if="authSubmitting" :size="16" class="spin" />
        <LogIn v-else :size="16" />
        <span>{{ authSubmitting ? '提交中' : (authMode === 'login' ? '登录' : '注册') }}</span>
      </button>
    </form>
  </div>

  <div v-else class="shell">
    <aside class="sidebar">
      <div class="sidebar__header">
        <div>
          <div class="eyebrow">Enterprise Knowledge System</div>
          <h1>文档与聊天</h1>
        </div>
        <button class="icon-button" type="button" @click="refreshDocs" :disabled="refreshingDocs">
          <RefreshCw :size="16" :class="{ spin: refreshingDocs }" />
        </button>
      </div>

      <section class="sidebar-section upload-panel">
        <div class="section-title">
          <Upload :size="16" />
          <span>上传文档</span>
        </div>
        <div class="upload-actions">
          <button class="primary-button" type="button" @click="openPicker" :disabled="uploadBusy">
            <Paperclip :size="16" />
            <span>{{ uploadBusy ? '上传中' : '选择文件' }}</span>
          </button>
          <input ref="fileInput" class="sr-only" type="file" @change="handleUpload" />
        </div>
        <p class="hint">支持 PDF、TXT、MD、DOCX、PPTX、XLSX 等常见文档。</p>
        <p v-if="uploadError" class="inline-error">
          <CircleAlert :size="14" />
          <span>{{ uploadError }}</span>
        </p>
      </section>

      <section class="sidebar-section">
        <div class="section-title">
          <FileText :size="16" />
          <span>已上传文档</span>
          <span class="count-badge">{{ docs.length }}</span>
        </div>

        <div v-if="docsLoading" class="empty-state compact">
          <Loader2 :size="18" class="spin" />
          <span>正在加载文档...</span>
        </div>

        <div v-else class="doc-list">
          <div
            v-for="doc in docs"
            :key="doc.document_id"
            class="doc-row"
            :class="{ active: selectedDocId === doc.document_id }"
            @click="selectDocument(doc)"
          >
            <button
              type="button"
              class="doc-row__select"
              @click.stop="selectDocument(doc)"
            >
              <div class="doc-row__main">
              <div class="doc-row__title">{{ doc.original_filename }}</div>
              <div class="doc-row__meta">
                <span>{{ doc.file_type?.toUpperCase() }}</span>
                <span>{{ formatBytes(doc.file_size) }}</span>
                <span :class="['status-pill', statusClass(doc.status)]">{{ statusLabel(doc.status) }}</span>
              </div>
              </div>
            </button>
            <button
              class="icon-button danger"
              type="button"
              :title="doc.status === 'ready' ? '删除文档' : '文档处理中，暂不能删除'"
              :disabled="doc.status !== 'ready'"
              @click.stop="handleDelete(doc)"
            >
              <Trash2 :size="14" />
            </button>
          </div>
        </div>

        <div v-if="selectedDoc" class="doc-detail">
          <div class="doc-detail__title">文档详情</div>
          <div class="doc-detail__row"><span>原名</span><strong>{{ selectedDoc.original_filename }}</strong></div>
          <div class="doc-detail__row"><span>状态</span><strong>{{ statusLabel(selectedDoc.status) }}</strong></div>
          <div class="doc-detail__row"><span>切块</span><strong>{{ selectedDoc.chunk_count }}</strong></div>
          <div class="doc-detail__row"><span>创建</span><strong>{{ formatDate(selectedDoc.created_at) }}</strong></div>
          <div class="doc-detail__row"><span>更新</span><strong>{{ formatDate(selectedDoc.updated_at) }}</strong></div>
          <div v-if="selectedDoc.error_message" class="inline-error">
            <CircleAlert :size="14" />
            <span>{{ selectedDoc.error_message }}</span>
          </div>
        </div>
      </section>
    </aside>

    <main class="workspace">
      <header class="topbar">
        <div class="topbar__left">
          <div class="brand-mark">
            <Bot :size="18" />
          </div>
          <div>
            <div class="eyebrow">ChatGPT 风格对话</div>
            <h2>知识库问答台</h2>
          </div>
        </div>

        <div class="toolbar">
          <label class="switch">
            <input v-model="streamMode" type="checkbox" />
            <span class="switch__track">
              <span class="switch__thumb"></span>
            </span>
            <span class="switch__label">{{ streamMode ? '流式输出' : '非流式输出' }}</span>
          </label>

          <label class="switch">
            <input v-model="useRetrieval" type="checkbox" />
            <span class="switch__track">
              <span class="switch__thumb"></span>
            </span>
            <span class="switch__label">知识检索</span>
          </label>

          <button
            class="ghost-button"
            type="button"
            :disabled="conversationLimitReached"
            :title="conversationLimitReached ? `已达到后端对话上限 ${maxConversations}` : '新建对话'"
            @click="resetChat"
          >
            <Plus :size="16" />
            <span>新对话</span>
          </button>

          <div
            class="profile-menu"
            v-if="currentUser"
            @mouseenter="openProfileMenu"
            @mouseleave="scheduleCloseProfileMenu"
            @focusin="openProfileMenu"
            @focusout="scheduleCloseProfileMenu"
          >
            <button
              class="ghost-button profile-menu__button"
              type="button"
              aria-haspopup="menu"
              :aria-expanded="profileMenuOpen"
              @click="openProfileMenu"
            >
              <span>个人信息</span>
              <ChevronDown :size="14" class="profile-menu__chevron" />
            </button>
            <div v-show="profileMenuOpen" class="profile-menu__panel" role="menu" @mouseenter="openProfileMenu" @mouseleave="scheduleCloseProfileMenu">
              <div class="profile-menu__account">
                <span class="profile-menu__account-label">账号</span>
                <span class="profile-menu__account-value">{{ currentUser.username }}</span>
              </div>
              <button class="profile-menu__item" type="button" @click="openPasswordModal">
                <KeyRound :size="15" />
                <span>修改密码</span>
              </button>
              <button class="profile-menu__item danger" type="button" @click="handleLogout">
                <LogOut :size="15" />
                <span>登出账号</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      <div class="workspace-body" :class="{ 'has-preview': selectedDocId }">
        <transition name="preview-panel">
          <section v-if="selectedDocId" class="preview-panel">
            <div class="preview-panel__header">
              <div>
                <div class="eyebrow">文档展开</div>
                <h3>{{ selectedDoc?.original_filename || '文档预览' }}</h3>
              </div>
              <button class="icon-button" type="button" title="关闭预览" @click="closeDocumentPreview">
                <ChevronDown :size="16" />
              </button>
            </div>

            <div v-if="selectedDoc" class="preview-panel__meta">
              <span>{{ selectedDoc.file_type?.toUpperCase() }}</span>
              <span>{{ statusLabel(selectedDoc.status) }}</span>
              <span>{{ selectedDoc.chunk_count }} chunks</span>
            </div>

            <div v-if="selectedDocLoading" class="preview-state">
              <Loader2 :size="18" class="spin" />
              <span>正在加载正文...</span>
            </div>

            <div v-else-if="selectedDocError" class="preview-state error">
              <CircleAlert :size="16" />
              <span>{{ selectedDocError }}</span>
            </div>

            <div v-else-if="spreadsheetPreview.length" class="spreadsheet-preview">
              <section v-for="sheet in spreadsheetPreview" :key="sheet.name" class="sheet-block">
                <div class="sheet-block__header">
                  <span class="sheet-block__name">{{ sheet.name }}</span>
                  <span class="sheet-block__meta">{{ sheet.headers.length }} 列 / {{ sheet.rows.length }} 行</span>
                </div>
                <div class="sheet-table-wrap" @pointerdown="startSheetDrag">
                  <table class="sheet-table">
                    <thead>
                      <tr>
                        <th v-for="(header, index) in sheet.headers" :key="`${sheet.name}-h-${index}`">
                          {{ header || `列${index + 1}` }}
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr v-for="(row, rowIndex) in sheet.rows" :key="`${sheet.name}-r-${rowIndex}`">
                        <td v-for="(cell, cellIndex) in sheet.headers" :key="`${sheet.name}-c-${rowIndex}-${cellIndex}`">
                          {{ row[cellIndex] || '' }}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </section>
            </div>
            <pre v-else class="preview-panel__content">{{ selectedDocContent || '暂无可预览内容' }}</pre>
          </section>
        </transition>

        <section class="chat-panel">
          <div ref="scrollHost" class="messages" @scroll="handleChatScroll">
            <article v-for="message in chatMessages" :key="message.id" class="message" :class="message.role">
              <div class="avatar">
                <component :is="message.role === 'user' ? User : Bot" :size="16" />
              </div>
              <div class="message__body">
                <div class="message__meta">
                  <span>{{ message.role === 'user' ? '你' : '助手' }}</span>
                  <span v-if="message.status === 'streaming'" class="typing-chip">
                    <Loader2 :size="12" class="spin" />
                    正在输出
                  </span>
                  <span v-else-if="message.status === 'thinking'" class="typing-chip">正在检索</span>
                  <span v-else-if="message.status === 'error'" class="typing-chip error">出错</span>
                </div>
                <div class="message__content" :class="{ empty: !message.content }">
                  <p v-if="message.content">{{ message.content }}</p>
                  <p v-else>...</p>
                </div>
                <div v-if="message.sources?.length" class="sources">
                  <button class="source-toggle" type="button" @click="toggleSources(message.id)">
                    <Wand2 :size="14" />
                    <span>来源</span>
                    <span class="source-toggle__count">{{ message.sources.length }}</span>
                    <ChevronDown :size="14" :class="{ open: expandedSources[message.id] }" />
                  </button>
                  <div v-if="expandedSources[message.id]" class="source-list">
                    <div v-for="(source, index) in message.sources" :key="index" class="source-item">
                      <div class="source-item__score">{{ Number(source.score || 0).toFixed(3) }}</div>
                      <div class="source-item__content">{{ source.content }}</div>
                    </div>
                  </div>
                </div>
              </div>
            </article>
          </div>

          <div class="composer">
            <textarea
              v-model="inputText"
              class="composer__input"
              placeholder="输入你的问题，Shift + Enter 换行，Enter 发送"
              rows="3"
              @keydown="handleComposerKeydown"
            />
            <div class="composer__footer">
              <div class="status-line">
                <span :class="['status-dot', chatBusy ? 'busy' : 'idle']"></span>
                <span>{{ chatBusy ? '正在生成回答' : '等待输入' }}</span>
              </div>
              <div class="composer-actions">
                <div ref="retrievalMenuRef" class="retrieval-menu composer-retrieval">
                  <button class="ghost-button retrieval-menu__button" type="button" @click.stop="toggleRetrievalMenu">
                    <span>{{ currentRetrievalLabel }}</span>
                    <ChevronDown :size="16" :class="{ open: retrievalMenuOpen }" />
                  </button>

                  <transition name="retrieval-menu">
                    <div v-if="retrievalMenuOpen" class="retrieval-menu__panel">
                      <button
                        v-for="option in retrievalOptions"
                        :key="option.value"
                        type="button"
                        class="retrieval-menu__item"
                        :class="{ active: retrievalMethod === option.value }"
                        @click.stop="chooseRetrievalMethod(option.value)"
                      >
                        <span>{{ option.label }}</span>
                        <Check v-if="retrievalMethod === option.value" :size="14" />
                      </button>
                    </div>
                  </transition>
                </div>

                <button class="primary-button send-button" type="button" @click="sendMessage" :disabled="chatBusy || !inputText.trim()">
                  <Send :size="16" />
                  <span>发送</span>
                  <ArrowRight :size="16" />
                </button>
              </div>
            </div>
          </div>
        </section>

        <aside class="right-rail">
          <section class="conversation-panel">
            <div class="conversation-panel__header">
              <div>
                <div class="eyebrow">Conversation</div>
                <h3>对话列表</h3>
              </div>
              <span class="count-badge">{{ conversations.length }} / {{ maxConversations }}</span>
            </div>

            <div class="conversation-list">
              <div
                v-for="conversation in conversationList"
                :key="conversation.conversation_id"
                class="conversation-row"
                :class="{ active: conversation.conversation_id === activeConversationId }"
              >
                <button
                  type="button"
                  class="conversation-row__select"
                  @click="activateConversation(conversation.conversation_id)"
                >
                  <div class="conversation-row__icon">
                    <MessageSquare :size="15" />
                  </div>
                  <div class="conversation-row__body">
                    <div class="conversation-row__title">{{ getConversationTitle(conversation) }}</div>
                    <div class="conversation-row__preview">{{ getConversationPreview(conversation) }}</div>
                    <div class="conversation-row__meta">
                      <span>{{ getConversationTurnCount(conversation) }} 轮</span>
                      <span>{{ formatDate(conversation.updated_at) }}</span>
                      <span v-if="isConversationStreaming(conversation.conversation_id)" class="conversation-row__status">
                        输出中
                      </span>
                    </div>
                    <div class="conversation-row__id">ID {{ conversation.conversation_id }}</div>
                  </div>
                </button>
                <button
                  class="icon-button danger"
                  type="button"
                  :title="isConversationStreaming(conversation.conversation_id) ? '输出中，暂不能删除' : '删除对话并清除长期记忆'"
                  :disabled="Boolean(deletingConversationIds[conversation.conversation_id]) || isConversationStreaming(conversation.conversation_id)"
                  @click.stop="removeConversation(conversation)"
                >
                  <Loader2 v-if="deletingConversationIds[conversation.conversation_id]" :size="14" class="spin" />
                  <Trash2 v-else :size="14" />
                </button>
              </div>
            </div>
          </section>

          <section class="summary-panel">
            <div class="summary-panel__header">
              <div>
                <div class="eyebrow">历史摘要</div>
                <h3>对话重点</h3>
              </div>
            </div>

            <div class="summary-panel__body">
              <div class="summary-block">
                <div class="summary-block__label">当前主题</div>
                <div class="summary-block__value">{{ conversationSummary.currentFocus || '暂无对话内容' }}</div>
              </div>

              <div class="summary-block">
                <div class="summary-block__label">最近提问</div>
                <ul v-if="conversationSummary.recentQuestions.length" class="summary-list">
                  <li v-for="(item, index) in conversationSummary.recentQuestions" :key="index">
                    {{ item }}
                  </li>
                </ul>
                <div v-else class="summary-empty">还没有用户提问。</div>
              </div>

              <div class="summary-block">
                <div class="summary-block__label">回答要点</div>
                <ul v-if="conversationSummary.keyPoints.length" class="summary-list">
                  <li v-for="(item, index) in conversationSummary.keyPoints" :key="index">
                    {{ item }}
                  </li>
                </ul>
                <div v-else class="summary-empty">等待助手生成重点摘要。</div>
              </div>
            </div>

            <div class="summary-footer">
              <span>轮次</span>
              <strong>{{ conversationSummary.turnCount }}</strong>
            </div>
          </section>
        </aside>
      </div>

      <div v-if="errorText" class="toast">
        <CircleAlert :size="16" />
        <span>{{ errorText }}</span>
        <button class="icon-button" type="button" @click="errorText = ''">
          <ChevronDown :size="14" />
        </button>
      </div>

      <transition name="modal">
        <div v-if="passwordModalOpen" class="modal-overlay" @click.self="closePasswordModal">
          <form class="modal-panel" @submit.prevent="handleChangePassword">
            <div class="modal-panel__header">
              <div>
                <div class="eyebrow">账户</div>
                <h3>修改密码</h3>
              </div>
              <button class="icon-button" type="button" title="关闭" @click="closePasswordModal">
                <X :size="16" />
              </button>
            </div>

            <label class="field">
              <span>当前密码</span>
              <div class="field__input">
                <input
                  v-model="passwordForm.currentPassword"
                  type="password"
                  autocomplete="current-password"
                  placeholder="请输入当前密码"
                />
              </div>
            </label>

            <label class="field">
              <span>新密码</span>
              <div class="field__input">
                <input
                  v-model="passwordForm.newPassword"
                  type="password"
                  autocomplete="new-password"
                  placeholder="请输入新密码"
                />
              </div>
            </label>

            <label class="field">
              <span>确认新密码</span>
              <div class="field__input">
                <input
                  v-model="passwordForm.confirmPassword"
                  type="password"
                  autocomplete="new-password"
                  placeholder="再次输入新密码"
                />
              </div>
            </label>

            <p v-if="passwordError" class="inline-error auth-error">
              <CircleAlert :size="14" />
              <span>{{ passwordError }}</span>
            </p>

            <button class="primary-button auth-button" type="submit" :disabled="passwordSubmitting">
              <Loader2 v-if="passwordSubmitting" :size="16" class="spin" />
              <KeyRound v-else :size="16" />
              <span>{{ passwordSubmitting ? '保存中' : '保存修改' }}</span>
            </button>
          </form>
        </div>
      </transition>
    </main>
  </div>
</template>
