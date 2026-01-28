import Foundation

public enum SigilBonjour {
    // v0: internal-only, subject to rename.
    public static let gatewayServiceType = "_sigil-gw._tcp"
    public static let gatewayServiceDomain = "local."
    public static let wideAreaGatewayServiceDomain = "sigil.internal."

    public static let gatewayServiceDomains = [
        gatewayServiceDomain,
        wideAreaGatewayServiceDomain,
    ]

    public static func normalizeServiceDomain(_ raw: String?) -> String {
        let trimmed = (raw ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            return self.gatewayServiceDomain
        }

        let lower = trimmed.lowercased()
        if lower == "local" || lower == "local." {
            return self.gatewayServiceDomain
        }

        return lower.hasSuffix(".") ? lower : (lower + ".")
    }
}
