import Foundation

public enum SigilLocationMode: String, Codable, Sendable, CaseIterable {
    case off
    case whileUsing
    case always
}
