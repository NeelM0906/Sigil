// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "SigilKit",
    platforms: [
        .iOS(.v18),
        .macOS(.v15),
    ],
    products: [
        .library(name: "SigilProtocol", targets: ["SigilProtocol"]),
        .library(name: "SigilKit", targets: ["SigilKit"]),
        .library(name: "SigilChatUI", targets: ["SigilChatUI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/steipete/ElevenLabsKit", exact: "0.1.0"),
        .package(url: "https://github.com/gonzalezreal/textual", exact: "0.3.1"),
    ],
    targets: [
        .target(
            name: "SigilProtocol",
            path: "Sources/SigilProtocol",
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency"),
            ]),
        .target(
            name: "SigilKit",
            path: "Sources/SigilKit",
            dependencies: [
                "SigilProtocol",
                .product(name: "ElevenLabsKit", package: "ElevenLabsKit"),
            ],
            resources: [
                .process("Resources"),
            ],
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency"),
            ]),
        .target(
            name: "SigilChatUI",
            path: "Sources/SigilChatUI",
            dependencies: [
                "SigilKit",
                .product(
                    name: "Textual",
                    package: "textual",
                    condition: .when(platforms: [.macOS, .iOS])),
            ],
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency"),
            ]),
        .testTarget(
            name: "SigilKitTests",
            dependencies: ["SigilKit", "SigilChatUI"],
            path: "Tests/SigilKitTests",
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency"),
                .enableExperimentalFeature("SwiftTesting"),
            ]),
    ])
