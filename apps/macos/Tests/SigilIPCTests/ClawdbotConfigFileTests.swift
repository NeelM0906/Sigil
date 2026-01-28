import Foundation
import Testing
@testable import Sigil

@Suite(.serialized)
struct SigilConfigFileTests {
    @Test
    func configPathRespectsEnvOverride() async {
        let override = FileManager().temporaryDirectory
            .appendingPathComponent("sigil-config-\(UUID().uuidString)")
            .appendingPathComponent("sigil.json")
            .path

        await TestIsolation.withEnvValues(["SIGIL_CONFIG_PATH": override]) {
            #expect(SigilConfigFile.url().path == override)
        }
    }

    @MainActor
    @Test
    func remoteGatewayPortParsesAndMatchesHost() async {
        let override = FileManager().temporaryDirectory
            .appendingPathComponent("sigil-config-\(UUID().uuidString)")
            .appendingPathComponent("sigil.json")
            .path

        await TestIsolation.withEnvValues(["SIGIL_CONFIG_PATH": override]) {
            SigilConfigFile.saveDict([
                "gateway": [
                    "remote": [
                        "url": "ws://gateway.ts.net:19999",
                    ],
                ],
            ])
            #expect(SigilConfigFile.remoteGatewayPort() == 19999)
            #expect(SigilConfigFile.remoteGatewayPort(matchingHost: "gateway.ts.net") == 19999)
            #expect(SigilConfigFile.remoteGatewayPort(matchingHost: "gateway") == 19999)
            #expect(SigilConfigFile.remoteGatewayPort(matchingHost: "other.ts.net") == nil)
        }
    }

    @MainActor
    @Test
    func setRemoteGatewayUrlPreservesScheme() async {
        let override = FileManager().temporaryDirectory
            .appendingPathComponent("sigil-config-\(UUID().uuidString)")
            .appendingPathComponent("sigil.json")
            .path

        await TestIsolation.withEnvValues(["SIGIL_CONFIG_PATH": override]) {
            SigilConfigFile.saveDict([
                "gateway": [
                    "remote": [
                        "url": "wss://old-host:111",
                    ],
                ],
            ])
            SigilConfigFile.setRemoteGatewayUrl(host: "new-host", port: 2222)
            let root = SigilConfigFile.loadDict()
            let url = ((root["gateway"] as? [String: Any])?["remote"] as? [String: Any])?["url"] as? String
            #expect(url == "wss://new-host:2222")
        }
    }

    @Test
    func stateDirOverrideSetsConfigPath() async {
        let dir = FileManager().temporaryDirectory
            .appendingPathComponent("sigil-state-\(UUID().uuidString)", isDirectory: true)
            .path

        await TestIsolation.withEnvValues([
            "SIGIL_CONFIG_PATH": nil,
            "SIGIL_STATE_DIR": dir,
        ]) {
            #expect(SigilConfigFile.stateDirURL().path == dir)
            #expect(SigilConfigFile.url().path == "\(dir)/sigil.json")
        }
    }
}
