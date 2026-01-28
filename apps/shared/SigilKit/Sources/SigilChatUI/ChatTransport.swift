import Foundation

public enum SigilChatTransportEvent: Sendable {
    case health(ok: Bool)
    case tick
    case chat(SigilChatEventPayload)
    case agent(SigilAgentEventPayload)
    case seqGap
}

public protocol SigilChatTransport: Sendable {
    func requestHistory(sessionKey: String) async throws -> SigilChatHistoryPayload
    func sendMessage(
        sessionKey: String,
        message: String,
        thinking: String,
        idempotencyKey: String,
        attachments: [SigilChatAttachmentPayload]) async throws -> SigilChatSendResponse

    func abortRun(sessionKey: String, runId: String) async throws
    func listSessions(limit: Int?) async throws -> SigilChatSessionsListResponse

    func requestHealth(timeoutMs: Int) async throws -> Bool
    func events() -> AsyncStream<SigilChatTransportEvent>

    func setActiveSessionKey(_ sessionKey: String) async throws
}

extension SigilChatTransport {
    public func setActiveSessionKey(_: String) async throws {}

    public func abortRun(sessionKey _: String, runId _: String) async throws {
        throw NSError(
            domain: "SigilChatTransport",
            code: 0,
            userInfo: [NSLocalizedDescriptionKey: "chat.abort not supported by this transport"])
    }

    public func listSessions(limit _: Int?) async throws -> SigilChatSessionsListResponse {
        throw NSError(
            domain: "SigilChatTransport",
            code: 0,
            userInfo: [NSLocalizedDescriptionKey: "sessions.list not supported by this transport"])
    }
}
