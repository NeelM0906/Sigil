import Testing
@testable import Sigil

@Suite(.serialized)
@MainActor
struct OnboardingCoverageTests {
    @Test func exerciseOnboardingPages() {
        OnboardingView.exerciseForTesting()
    }
}
