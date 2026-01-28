// swift-tools-version: 6.2
// Package manifest for the Sigil macOS companion (menu bar app + IPC library).

import PackageDescription

let package = Package(
    name: "Sigil",
    platforms: [
        .macOS(.v15),
    ],
    products: [
        .library(name: "SigilIPC", targets: ["SigilIPC"]),
        .library(name: "SigilDiscovery", targets: ["SigilDiscovery"]),
        .executable(name: "Sigil", targets: ["Sigil"]),
        .executable(name: "sigil-mac", targets: ["SigilMacCLI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/orchetect/MenuBarExtraAccess", exact: "1.2.2"),
        .package(url: "https://github.com/swiftlang/swift-subprocess.git", from: "0.1.0"),
        .package(url: "https://github.com/apple/swift-log.git", from: "1.8.0"),
        .package(url: "https://github.com/sparkle-project/Sparkle", from: "2.8.1"),
        .package(url: "https://github.com/steipete/Peekaboo.git", branch: "main"),
        .package(path: "../shared/SigilKit"),
        .package(path: "../../Swabble"),
    ],
    targets: [
        .target(
            name: "SigilIPC",
            dependencies: [],
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency"),
            ]),
        .target(
            name: "SigilDiscovery",
            dependencies: [
                .product(name: "SigilKit", package: "SigilKit"),
            ],
            path: "Sources/SigilDiscovery",
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency"),
            ]),
        .executableTarget(
            name: "Sigil",
            dependencies: [
                "SigilIPC",
                "SigilDiscovery",
                .product(name: "SigilKit", package: "SigilKit"),
                .product(name: "SigilChatUI", package: "SigilKit"),
                .product(name: "SigilProtocol", package: "SigilKit"),
                .product(name: "SwabbleKit", package: "swabble"),
                .product(name: "MenuBarExtraAccess", package: "MenuBarExtraAccess"),
                .product(name: "Subprocess", package: "swift-subprocess"),
                .product(name: "Logging", package: "swift-log"),
                .product(name: "Sparkle", package: "Sparkle"),
                .product(name: "PeekabooBridge", package: "Peekaboo"),
                .product(name: "PeekabooAutomationKit", package: "Peekaboo"),
            ],
            exclude: [
                "Resources/Info.plist",
            ],
            resources: [
                .copy("Resources/Sigil.icns"),
                .copy("Resources/DeviceModels"),
            ],
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency"),
            ]),
        .executableTarget(
            name: "SigilMacCLI",
            dependencies: [
                "SigilDiscovery",
                .product(name: "SigilKit", package: "SigilKit"),
                .product(name: "SigilProtocol", package: "SigilKit"),
            ],
            path: "Sources/SigilMacCLI",
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency"),
            ]),
        .testTarget(
            name: "SigilIPCTests",
            dependencies: [
                "SigilIPC",
                "Sigil",
                "SigilDiscovery",
                .product(name: "SigilProtocol", package: "SigilKit"),
                .product(name: "SwabbleKit", package: "swabble"),
            ],
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency"),
                .enableExperimentalFeature("SwiftTesting"),
            ]),
    ])
