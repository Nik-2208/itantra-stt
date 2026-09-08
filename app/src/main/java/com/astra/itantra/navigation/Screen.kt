package com.astra.itantra.navigation

sealed class Screen(val route: String) {
    object Splash : Screen("splash")
    object LanguageSelection : Screen("language_selection")
    object Permissions : Screen("permissions")
    object Home : Screen("home")
    object DeviceDiscovery : Screen("device_discovery")
    object WalkieTalkie : Screen("walkie_talkie")
    object TranscriptHistory : Screen("transcript_history")
    object AlertMode : Screen("alert_mode")
    object Settings : Screen("settings")
    object ConnectionDetails : Screen("connection_details")
    object About : Screen("about")
}
