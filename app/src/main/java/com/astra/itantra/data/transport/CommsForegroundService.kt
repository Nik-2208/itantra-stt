package com.astra.itantra.data.transport

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import com.astra.itantra.R
import dagger.hilt.android.AndroidEntryPoint

@AndroidEntryPoint
class CommsForegroundService : Service() {

    override fun onCreate() {
        super.onCreate()
        startForegroundServiceNotification()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun startForegroundServiceNotification() {
        try {
            val channelId = "itantra_comms_channel"
            val channelName = "iTantra Offline Walkie-Talkie Service"
            val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                val channel = NotificationChannel(
                    channelId,
                    channelName,
                    NotificationManager.IMPORTANCE_LOW
                )
                manager.createNotificationChannel(channel)
            }

            val notification: Notification = NotificationCompat.Builder(this, channelId)
                .setContentTitle("iTantra Active")
                .setContentText("Offline peer mesh & audio listener running")
                .setSmallIcon(R.mipmap.ic_launcher)
                .setOngoing(true)
                .build()

            startForeground(1001, notification)
        } catch (e: Exception) {
            // Permission pending or not granted yet
            e.printStackTrace()
        }
    }

    companion object {
        fun startService(context: Context) {
            try {
                val intent = Intent(context, CommsForegroundService::class.java)
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    context.startForegroundService(intent)
                } else {
                    context.startService(intent)
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }

        fun stopService(context: Context) {
            try {
                val intent = Intent(context, CommsForegroundService::class.java)
                context.stopService(intent)
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }
}
