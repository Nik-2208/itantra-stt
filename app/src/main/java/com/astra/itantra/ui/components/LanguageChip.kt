package com.astra.itantra.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.astra.itantra.data.model.Language
import com.astra.itantra.ui.theme.DeepIndigoPrimary
import com.astra.itantra.ui.theme.WarmSaffronAccent

@Composable
fun LanguageChip(
    language: Language,
    isSelected: Boolean,
    onClick: () -> Unit
) {
    val bgColor = if (isSelected) DeepIndigoPrimary else Color.LightGray.copy(alpha = 0.3f)
    val textColor = if (isSelected) WarmSaffronAccent else MaterialTheme.colorScheme.onSurface

    Box(
        modifier = Modifier
            .background(bgColor, RoundedCornerShape(16.dp))
            .clickable { onClick() }
            .padding(horizontal = 14.dp, vertical = 8.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                text = language.nativeName,
                style = MaterialTheme.typography.bodyMedium,
                color = textColor
            )
            Spacer(modifier = Modifier.width(4.dp))
            Text(
                text = "(${language.englishName})",
                style = MaterialTheme.typography.labelSmall,
                color = textColor.copy(alpha = 0.7f)
            )
        }
    }
}
