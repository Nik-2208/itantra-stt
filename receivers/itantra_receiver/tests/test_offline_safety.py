"""
Network Safety Verification Test (test_offline_safety.py)
Validates that the entire receiver pipeline runs strictly offline with ZERO network activity.
Monkey-patches socket.socket.connect and socket.create_connection to trap accidental egress.
"""

import socket
import unittest
from formatter import ReceiverFormatter
from pipeline import ReceiverPipeline
from translation import OnDemandTranslator, MockTranslationAdapter
from tts import OfflineTTS, MockTTSAdapter


class NetworkAccessDetectedError(Exception):
    """Raised if any pipeline code attempts outbound network I/O."""
    pass


class TestOfflineSafety(unittest.TestCase):
    def setUp(self):
        # Save original socket methods
        self._orig_socket_connect = socket.socket.connect
        self._orig_create_connection = socket.create_connection

        # Guard function that raises error on any network connection attempt
        def blocked_connect(*args, **kwargs):
            raise NetworkAccessDetectedError(
                f"ILLEGAL NETWORK CALL DETECTED: socket connect attempted with args={args}, kwargs={kwargs}. "
                f"The iTantra Receiver Pipeline must operate 100% offline."
            )

        # Apply monkey patch
        socket.socket.connect = blocked_connect
        socket.create_connection = blocked_connect

        self.pipeline = ReceiverPipeline(
            translator=OnDemandTranslator(adapter=MockTranslationAdapter()),
            tts=OfflineTTS(adapter=MockTTSAdapter()),
            formatter=ReceiverFormatter(),
        )

    def tearDown(self):
        # Restore socket methods
        socket.socket.connect = self._orig_socket_connect
        socket.create_connection = self._orig_create_connection

    def test_pipeline_runs_without_network(self):
        """Ensure full end-to-end execution makes zero socket/network calls."""
        try:
            result = self.pipeline.translate_and_speak(
                text="यहाँ आपातकालीन स्थिति है, तुरंत मदद भेजें।",
                source_language="hi",
                target_language="en",
            )
            self.assertIsNotNone(result)
            self.assertIn("audio", result)
            self.assertIn("formatted_output", result)
        except NetworkAccessDetectedError as e:
            self.fail(f"Pipeline violated offline requirement: {e}")


if __name__ == "__main__":
    unittest.main()
