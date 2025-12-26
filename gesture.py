"""
Workshop Gesture Interface (Phase 2)
Hand tracking and gesture recognition using MediaPipe

This is a skeleton for Phase 2 - implement when ready to add gesture control.
"""

from typing import Optional, Callable, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio


class GestureType(Enum):
    """Recognized gesture types"""
    NONE = "none"
    POINT = "point"              # Index finger extended
    PINCH = "pinch"              # Thumb and index together
    GRAB = "grab"                # Closed fist
    OPEN_PALM = "open_palm"      # All fingers extended
    THUMBS_UP = "thumbs_up"      # Thumbs up gesture
    SWIPE_LEFT = "swipe_left"    # Quick left motion
    SWIPE_RIGHT = "swipe_right"  # Quick right motion
    SWIPE_UP = "swipe_up"        # Quick up motion
    SWIPE_DOWN = "swipe_down"    # Quick down motion
    CIRCLE = "circle"            # Circular motion
    TWO_FINGER_PINCH = "two_finger_pinch"  # Zoom gesture


@dataclass
class HandState:
    """Current state of detected hand"""
    detected: bool = False
    position: Tuple[float, float, float] = (0, 0, 0)  # x, y, z normalized
    gesture: GestureType = GestureType.NONE
    confidence: float = 0.0
    landmarks: list = None  # MediaPipe landmarks


@dataclass
class GestureEvent:
    """A gesture event with context"""
    gesture: GestureType
    hand: str  # "left" or "right"
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    duration: float  # seconds held
    confidence: float


class GestureInterface:
    """
    Hand tracking and gesture recognition.
    
    Usage:
        gesture = GestureInterface()
        gesture.on_gesture(GestureType.PINCH, handle_pinch)
        await gesture.start()
    """
    
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self._running = False
        self._handlers: Dict[GestureType, list] = {}
        self._hands = None
        self._cap = None
        
        # State tracking
        self._left_hand = HandState()
        self._right_hand = HandState()
        self._gesture_start_times: Dict[str, float] = {}
    
    def _init_mediapipe(self) -> bool:
        """Initialize MediaPipe hands"""
        try:
            import mediapipe as mp
            
            self._mp_hands = mp.solutions.hands
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            self._mp_draw = mp.solutions.drawing_utils
            return True
            
        except ImportError:
            print("⚠️  MediaPipe not installed.")
            print("   Install with: pip install mediapipe")
            return False
    
    def _init_camera(self) -> bool:
        """Initialize camera capture"""
        try:
            import cv2
            self._cap = cv2.VideoCapture(self.camera_index)
            
            if not self._cap.isOpened():
                print(f"⚠️  Cannot open camera {self.camera_index}")
                return False
            
            # Set resolution
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self._cap.set(cv2.CAP_PROP_FPS, 30)
            
            return True
            
        except ImportError:
            print("⚠️  OpenCV not installed.")
            print("   Install with: pip install opencv-python")
            return False
    
    def on_gesture(self, gesture: GestureType, handler: Callable[[GestureEvent], None]):
        """Register a handler for a gesture type"""
        if gesture not in self._handlers:
            self._handlers[gesture] = []
        self._handlers[gesture].append(handler)
    
    async def start(self, show_preview: bool = False):
        """Start gesture recognition loop"""
        if not self._init_mediapipe() or not self._init_camera():
            return
        
        import cv2
        
        self._running = True
        print("👋 Gesture recognition started. Press 'q' in preview to quit.")
        
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                await asyncio.sleep(0.01)
                continue
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._hands.process(rgb)
            
            # Process detected hands
            if results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Determine handedness
                    handedness = results.multi_handedness[i].classification[0].label
                    
                    # Update state
                    hand_state = self._process_hand(hand_landmarks, handedness)
                    
                    if handedness == "Left":
                        self._left_hand = hand_state
                    else:
                        self._right_hand = hand_state
                    
                    # Check for gesture events
                    await self._check_gestures(hand_state, handedness.lower())
                    
                    # Draw landmarks if preview enabled
                    if show_preview:
                        self._mp_draw.draw_landmarks(
                            frame, hand_landmarks, 
                            self._mp_hands.HAND_CONNECTIONS
                        )
            else:
                # No hands detected
                self._left_hand.detected = False
                self._right_hand.detected = False
            
            # Show preview
            if show_preview:
                # Add gesture text
                cv2.putText(
                    frame, 
                    f"L: {self._left_hand.gesture.value} R: {self._right_hand.gesture.value}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                
                cv2.imshow("Workshop Gesture Preview", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Yield control
            await asyncio.sleep(0.01)
        
        self.cleanup()
    
    def _process_hand(self, landmarks, handedness: str) -> HandState:
        """Process hand landmarks to determine state and gesture"""
        state = HandState()
        state.detected = True
        state.landmarks = landmarks.landmark
        
        # Get wrist position (landmark 0)
        wrist = landmarks.landmark[0]
        state.position = (wrist.x, wrist.y, wrist.z)
        
        # Detect gesture
        state.gesture = self._classify_gesture(landmarks.landmark)
        state.confidence = 0.9  # TODO: Calculate actual confidence
        
        return state
    
    def _classify_gesture(self, landmarks) -> GestureType:
        """Classify the current hand pose as a gesture"""
        # Get key landmarks
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        index_mcp = landmarks[5]
        middle_mcp = landmarks[9]
        ring_mcp = landmarks[13]
        pinky_mcp = landmarks[17]
        
        # Helper: check if finger is extended
        def is_extended(tip, mcp):
            return tip.y < mcp.y  # tip is above mcp (screen coords)
        
        # Helper: distance between two points
        def distance(p1, p2):
            return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5
        
        # Check pinch (thumb and index close)
        if distance(thumb_tip, index_tip) < 0.05:
            return GestureType.PINCH
        
        # Check point (only index extended)
        index_ext = is_extended(index_tip, index_mcp)
        middle_ext = is_extended(middle_tip, middle_mcp)
        ring_ext = is_extended(ring_tip, ring_mcp)
        pinky_ext = is_extended(pinky_tip, pinky_mcp)
        
        if index_ext and not middle_ext and not ring_ext and not pinky_ext:
            return GestureType.POINT
        
        # Check open palm (all extended)
        if index_ext and middle_ext and ring_ext and pinky_ext:
            return GestureType.OPEN_PALM
        
        # Check grab/fist (none extended)
        if not index_ext and not middle_ext and not ring_ext and not pinky_ext:
            return GestureType.GRAB
        
        # TODO: Add more gestures (thumbs up, swipes, circles)
        
        return GestureType.NONE
    
    async def _check_gestures(self, hand_state: HandState, hand: str):
        """Check for gesture events and trigger handlers"""
        import time
        
        gesture = hand_state.gesture
        key = f"{hand}_{gesture.value}"
        
        if gesture != GestureType.NONE:
            # Track gesture start time
            if key not in self._gesture_start_times:
                self._gesture_start_times[key] = time.time()
            
            duration = time.time() - self._gesture_start_times[key]
            
            # Create event
            event = GestureEvent(
                gesture=gesture,
                hand=hand,
                position=hand_state.position,
                velocity=(0, 0, 0),  # TODO: Calculate velocity
                duration=duration,
                confidence=hand_state.confidence
            )
            
            # Trigger handlers
            if gesture in self._handlers:
                for handler in self._handlers[gesture]:
                    try:
                        result = handler(event)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        print(f"Gesture handler error: {e}")
        else:
            # Clear start time when gesture ends
            self._gesture_start_times = {
                k: v for k, v in self._gesture_start_times.items()
                if not k.startswith(hand)
            }
    
    def stop(self):
        """Stop gesture recognition"""
        self._running = False
    
    def cleanup(self):
        """Clean up resources"""
        self._running = False
        
        if self._cap:
            self._cap.release()
        
        try:
            import cv2
            cv2.destroyAllWindows()
        except:
            pass
    
    @property
    def left_hand(self) -> HandState:
        """Get current left hand state"""
        return self._left_hand
    
    @property
    def right_hand(self) -> HandState:
        """Get current right hand state"""
        return self._right_hand


# === Example Usage ===

async def example():
    """Example of gesture control integration"""
    
    def on_point(event: GestureEvent):
        print(f"Pointing at {event.position} with {event.hand} hand")
    
    def on_pinch(event: GestureEvent):
        if event.duration > 0.5:  # Held for 0.5s
            print(f"Pinch and hold at {event.position}")
    
    def on_palm(event: GestureEvent):
        print("Stop gesture detected!")
    
    # Create interface
    gesture = GestureInterface()
    
    # Register handlers
    gesture.on_gesture(GestureType.POINT, on_point)
    gesture.on_gesture(GestureType.PINCH, on_pinch)
    gesture.on_gesture(GestureType.OPEN_PALM, on_palm)
    
    # Start recognition with preview
    await gesture.start(show_preview=True)


if __name__ == "__main__":
    print("Gesture Module Test")
    print("Press 'q' in preview window to quit")
    asyncio.run(example())
