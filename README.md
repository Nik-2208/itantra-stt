# ASTRA – iTantra

> **Offline, multilingual, peer-to-peer speech translation using on-device AI.**

---

## Overview

iTantra is an Android application that enables two nearby users to communicate across different languages **without requiring an internet connection**.

The application uses:

- On-device Speech-to-Text (STT)
- On-device Machine Translation (MT)
- On-device Text-to-Speech (TTS)
- Nearby Connections API for peer-to-peer communication

No cloud services are required during normal operation.

---

## High-Level Architecture


Sender Device
```
────────────────────────────────

Microphone
│
▼
Audio Capture + VAD
│
▼
Speech-to-Text
│
▼
Text + Metadata
│
▼
Nearby Transport
═══════════════════════════════
Wi-Fi / Bluetooth
═══════════════════════════════
▼
Text + Metadata
│
▼
Translation
│
▼
Text-to-Speech
│
▼
Speaker

────────────────────────────────
Receiver Device
```

---

## Project Structure

```
iTantra/

app/            Main Android application
core/           Shared interfaces & models
audio/          Audio capture & Voice Activity Detection
stt/            Speech-to-Text engine
langgen/        Translation + Text-to-Speech
transport/      Nearby communication layer

gradle/
```

---

## Module Responsibilities

### app

- User Interface
- Dependency Injection
- Application State
- Playback
- Permissions
- Integration of all modules

---

### core

Contains shared contracts used by every module.

Examples:

- Data models
- Interfaces
- Enums
- Shared constants

No module should directly depend on another feature module.

All communication happens through `core`.

---

### audio

Responsible for:

- AudioRecord
- Microphone input
- Voice Activity Detection (VAD)
- PCM audio generation

Outputs speech chunks.

---

### stt

Responsible for:

- On-device Speech Recognition
- Language detection (if applicable)
- Producing text

Consumes PCM audio.

Outputs text.

---

### langgen

Responsible for:

- Machine Translation
- Text-to-Speech

Consumes text.

Outputs translated text and synthesized audio.

---

### transport

Responsible for:

- Nearby Connections
- Device discovery
- Pairing
- Reliable messaging
- Relay support (future)

Transmits text between devices.

---

## Branch Strategy

| Branch | Purpose |
|---------|---------|
| `main` | Stable integration branch |
| `feature/core` | Shared interfaces |
| `feature/audio` | Audio module |
| `feature/stt` | Speech Recognition |
| `feature/langgen` | Translation + TTS |
| `feature/transport` | Nearby communication |
| `feature/app` | UI & integration |

---

## Development Workflow

1. Pull latest changes.

```
git checkout main
git pull
```

2. Switch to your feature branch.

```
git checkout feature/<module>
```

3. Implement only your assigned module.

4. Commit frequently.

```
git add .
git commit -m "Describe changes"
```

5. Push your branch.

```
git push
```

6. Open a Pull Request.

---

## Rules

- Do **not** modify another module.
- Shared interfaces belong in `core`.
- Keep commits small and descriptive.
- Ensure the project builds before pushing.
- Resolve merge conflicts on your own branch.

---

## Build Requirements

| Requirement | Version |
|-------------|---------|
| Android Studio | Narwhal (or newer) |
| Android Gradle Plugin | 8.13.2 |
| Kotlin | 2.0.21 |
| Compile SDK | 36 |
| Target SDK | 36 |
| Minimum SDK | 24 |
| JDK | 11 |

---

## Build

Clone the repository.

```
git clone <repository-url>
```

Open the project in Android Studio.

Build using:

```
./gradlew build
```

Run:

```
./gradlew installDebug
```

---

## Coding Standards

- Kotlin only.
- Follow official Android Kotlin style.
- Prefer immutable data.
- Keep modules independent.
- Avoid circular dependencies.
- Document public APIs.

---

## Team Responsibilities

| Member | Module |
|---------|--------|
| Person 1 | Audio |
| Person 2 | STT |
| Person 3 | Translation & TTS |
| Person 4 | Transport |
| Person 5 | App/UI |
| Integration Lead | Core + Integration |

---

## Current Status

- [x] Multi-module Gradle project
- [x] Android project configured
- [x] Module separation complete
- [ ] Core interfaces
- [ ] Audio implementation
- [ ] STT implementation
- [ ] Translation implementation
- [ ] Transport implementation
- [ ] UI integration
- [ ] End-to-end testing
