import Foundation

public enum SigilCapability: String, Codable, Sendable {
    case canvas
    case camera
    case screen
    case voiceWake
    case location
}
