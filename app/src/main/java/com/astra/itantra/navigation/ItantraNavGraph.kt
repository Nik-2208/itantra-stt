package com.astra.itantra.navigation

import androidx.compose.runtime.Composable
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.astra.itantra.ui.screens.about.AboutScreen
import com.astra.itantra.ui.screens.alert.AlertModeScreen
import com.astra.itantra.ui.screens.connect.DeviceDiscoveryScreen
import com.astra.itantra.ui.screens.diagnostics.ConnectionDetailsScreen
import com.astra.itantra.ui.screens.history.TranscriptHistoryScreen
import com.astra.itantra.ui.screens.onboarding.LanguageSelectionScreen
import com.astra.itantra.ui.screens.onboarding.PermissionsScreen
import com.astra.itantra.ui.screens.settings.SettingsScreen
import com.astra.itantra.ui.screens.splash.SplashScreen
import com.astra.itantra.ui.screens.talk.WalkieTalkieScreen
import com.astra.itantra.ui.viewmodel.ConnectViewModel
import com.astra.itantra.ui.viewmodel.DiagnosticsViewModel
import com.astra.itantra.ui.viewmodel.HistoryViewModel
import com.astra.itantra.ui.viewmodel.SettingsViewModel
import com.astra.itantra.ui.viewmodel.WalkieTalkieViewModel

import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue

@Composable
fun ItantraNavGraph(
    navController: NavHostController = rememberNavController(),
    startDestination: String = Screen.Splash.route
) {
    NavHost(
        navController = navController,
        startDestination = startDestination
    ) {
        composable(Screen.Splash.route) {
            val settingsViewModel: SettingsViewModel = hiltViewModel()
            val isInitialized by settingsViewModel.isInitialized.collectAsState()

            SplashScreen(
                isAlreadyInitialized = isInitialized,
                onSplashFinished = {
                    navController.navigate(Screen.LanguageSelection.route) {
                        popUpTo(Screen.Splash.route) { inclusive = true }
                    }
                },
                onNavigateToMain = {
                    navController.navigate(Screen.DeviceDiscovery.route) {
                        popUpTo(Screen.Splash.route) { inclusive = true }
                    }
                }
            )
        }

        composable(Screen.LanguageSelection.route) {
            val settingsViewModel: SettingsViewModel = hiltViewModel()
            LanguageSelectionScreen(
                onLanguageSelected = { selectedLang ->
                    settingsViewModel.setSelectedLanguage(selectedLang)
                    navController.navigate(Screen.Permissions.route)
                }
            )
        }

        composable(Screen.Permissions.route) {
            val settingsViewModel: SettingsViewModel = hiltViewModel()
            PermissionsScreen(
                onPermissionsGranted = {
                    settingsViewModel.setInitialized(true)
                    navController.navigate(Screen.DeviceDiscovery.route) {
                        popUpTo(Screen.LanguageSelection.route) { inclusive = true }
                    }
                }
            )
        }

        composable(Screen.Home.route) {
            val viewModel: ConnectViewModel = hiltViewModel()
            DeviceDiscoveryScreen(
                viewModel = viewModel,
                onNavigateToComms = { navController.navigate(Screen.WalkieTalkie.route) },
                onNavigateToHistory = { navController.navigate(Screen.TranscriptHistory.route) },
                onNavigateToStats = { navController.navigate(Screen.ConnectionDetails.route) },
                onNavigateToAlert = { navController.navigate(Screen.AlertMode.route) },
                onNavigateToSettings = { navController.navigate(Screen.Settings.route) }
            )
        }

        composable(Screen.DeviceDiscovery.route) {
            val viewModel: ConnectViewModel = hiltViewModel()
            DeviceDiscoveryScreen(
                viewModel = viewModel,
                onNavigateToComms = { navController.navigate(Screen.WalkieTalkie.route) },
                onNavigateToHistory = { navController.navigate(Screen.TranscriptHistory.route) },
                onNavigateToStats = { navController.navigate(Screen.ConnectionDetails.route) },
                onNavigateToAlert = { navController.navigate(Screen.AlertMode.route) },
                onNavigateToSettings = { navController.navigate(Screen.Settings.route) }
            )
        }

        composable(Screen.WalkieTalkie.route) {
            val viewModel: WalkieTalkieViewModel = hiltViewModel()
            WalkieTalkieScreen(
                viewModel = viewModel,
                onNavigateBack = { navController.popBackStack() },
                onNavigateToDiscovery = { navController.navigate(Screen.DeviceDiscovery.route) },
                onNavigateToHistory = { navController.navigate(Screen.TranscriptHistory.route) },
                onNavigateToStats = { navController.navigate(Screen.ConnectionDetails.route) },
                onNavigateToAlert = { navController.navigate(Screen.AlertMode.route) }
            )
        }

        composable(Screen.TranscriptHistory.route) {
            val viewModel: HistoryViewModel = hiltViewModel()
            TranscriptHistoryScreen(
                viewModel = viewModel,
                onNavigateBack = { navController.popBackStack() },
                onNavigateToDiscovery = { navController.navigate(Screen.DeviceDiscovery.route) },
                onNavigateToComms = { navController.navigate(Screen.WalkieTalkie.route) },
                onNavigateToStats = { navController.navigate(Screen.ConnectionDetails.route) }
            )
        }

        composable(Screen.AlertMode.route) {
            val viewModel: WalkieTalkieViewModel = hiltViewModel()
            AlertModeScreen(
                viewModel = viewModel,
                onNavigateBack = { navController.popBackStack() }
            )
        }

        composable(Screen.Settings.route) {
            val viewModel: SettingsViewModel = hiltViewModel()
            SettingsScreen(
                viewModel = viewModel,
                onNavigateBack = { navController.popBackStack() }
            )
        }

        composable(Screen.ConnectionDetails.route) {
            val viewModel: DiagnosticsViewModel = hiltViewModel()
            ConnectionDetailsScreen(
                viewModel = viewModel,
                onNavigateBack = { navController.popBackStack() },
                onNavigateToDiscovery = { navController.navigate(Screen.DeviceDiscovery.route) },
                onNavigateToComms = { navController.navigate(Screen.WalkieTalkie.route) },
                onNavigateToHistory = { navController.navigate(Screen.TranscriptHistory.route) }
            )
        }

        composable(Screen.About.route) {
            AboutScreen(
                onNavigateBack = { navController.popBackStack() }
            )
        }
    }
}
