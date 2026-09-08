package com.astra.itantra.di

import android.content.Context
import androidx.room.Room
import com.astra.itantra.data.db.AppDatabase
import com.astra.itantra.data.db.MessageDao
import com.astra.itantra.data.stt.SttEngine
import com.astra.itantra.data.stt.TfliteSttEngine
import com.astra.itantra.data.transport.BluetoothTransport
import com.astra.itantra.data.transport.TransportManager
import com.astra.itantra.data.tts.TfliteTtsEngine
import com.astra.itantra.data.tts.TtsEngine
import dagger.Binds
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object DatabaseModule {

    @Provides
    @Singleton
    fun provideAppDatabase(@ApplicationContext context: Context): AppDatabase {
        return Room.databaseBuilder(
            context,
            AppDatabase::class.java,
            "itantra_database"
        ).fallbackToDestructiveMigration().build()
    }

    @Provides
    fun provideMessageDao(database: AppDatabase): MessageDao {
        return database.messageDao()
    }

    @Provides
    @Singleton
    fun provideApplicationContext(@ApplicationContext context: Context): Context {
        return context
    }
}

@Module
@InstallIn(SingletonComponent::class)
abstract class EngineModule {

    @Binds
    @Singleton
    abstract fun bindSttEngine(engine: TfliteSttEngine): SttEngine

    @Binds
    @Singleton
    abstract fun bindTtsEngine(engine: TfliteTtsEngine): TtsEngine

    @Binds
    @Singleton
    abstract fun bindTransportManager(transport: BluetoothTransport): TransportManager
}
