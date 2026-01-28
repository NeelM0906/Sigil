package com.sigil.android.protocol

import org.junit.Assert.assertEquals
import org.junit.Test

class SigilProtocolConstantsTest {
  @Test
  fun canvasCommandsUseStableStrings() {
    assertEquals("canvas.present", SigilCanvasCommand.Present.rawValue)
    assertEquals("canvas.hide", SigilCanvasCommand.Hide.rawValue)
    assertEquals("canvas.navigate", SigilCanvasCommand.Navigate.rawValue)
    assertEquals("canvas.eval", SigilCanvasCommand.Eval.rawValue)
    assertEquals("canvas.snapshot", SigilCanvasCommand.Snapshot.rawValue)
  }

  @Test
  fun a2uiCommandsUseStableStrings() {
    assertEquals("canvas.a2ui.push", SigilCanvasA2UICommand.Push.rawValue)
    assertEquals("canvas.a2ui.pushJSONL", SigilCanvasA2UICommand.PushJSONL.rawValue)
    assertEquals("canvas.a2ui.reset", SigilCanvasA2UICommand.Reset.rawValue)
  }

  @Test
  fun capabilitiesUseStableStrings() {
    assertEquals("canvas", SigilCapability.Canvas.rawValue)
    assertEquals("camera", SigilCapability.Camera.rawValue)
    assertEquals("screen", SigilCapability.Screen.rawValue)
    assertEquals("voiceWake", SigilCapability.VoiceWake.rawValue)
  }

  @Test
  fun screenCommandsUseStableStrings() {
    assertEquals("screen.record", SigilScreenCommand.Record.rawValue)
  }
}
