import Foundation

public enum SigilCameraCommand: String, Codable, Sendable {
    case list = "camera.list"
    case snap = "camera.snap"
    case clip = "camera.clip"
}

public enum SigilCameraFacing: String, Codable, Sendable {
    case back
    case front
}

public enum SigilCameraImageFormat: String, Codable, Sendable {
    case jpg
    case jpeg
}

public enum SigilCameraVideoFormat: String, Codable, Sendable {
    case mp4
}

public struct SigilCameraSnapParams: Codable, Sendable, Equatable {
    public var facing: SigilCameraFacing?
    public var maxWidth: Int?
    public var quality: Double?
    public var format: SigilCameraImageFormat?
    public var deviceId: String?
    public var delayMs: Int?

    public init(
        facing: SigilCameraFacing? = nil,
        maxWidth: Int? = nil,
        quality: Double? = nil,
        format: SigilCameraImageFormat? = nil,
        deviceId: String? = nil,
        delayMs: Int? = nil)
    {
        self.facing = facing
        self.maxWidth = maxWidth
        self.quality = quality
        self.format = format
        self.deviceId = deviceId
        self.delayMs = delayMs
    }
}

public struct SigilCameraClipParams: Codable, Sendable, Equatable {
    public var facing: SigilCameraFacing?
    public var durationMs: Int?
    public var includeAudio: Bool?
    public var format: SigilCameraVideoFormat?
    public var deviceId: String?

    public init(
        facing: SigilCameraFacing? = nil,
        durationMs: Int? = nil,
        includeAudio: Bool? = nil,
        format: SigilCameraVideoFormat? = nil,
        deviceId: String? = nil)
    {
        self.facing = facing
        self.durationMs = durationMs
        self.includeAudio = includeAudio
        self.format = format
        self.deviceId = deviceId
    }
}
