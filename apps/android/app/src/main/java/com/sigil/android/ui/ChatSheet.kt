package com.sigil.android.ui

import androidx.compose.runtime.Composable
import com.sigil.android.MainViewModel
import com.sigil.android.ui.chat.ChatSheetContent

@Composable
fun ChatSheet(viewModel: MainViewModel) {
  ChatSheetContent(viewModel = viewModel)
}
