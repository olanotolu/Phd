"""
Restaurant Booking Dialogue Data Collection and Formatting

Collects and formats restaurant booking dialogues for TLM training.
This creates the "normal" baseline that TLM will learn.
"""

import random
import sys
import os
from typing import List, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import text_to_indices, ALPHABET, ALPHABET_SIZE


def generate_restaurant_dialogues(n_dialogues: int = 200) -> List[str]:
    """
    Generate restaurant booking dialogue examples.
    These represent "normal" restaurant booking interactions.
    
    Args:
        n_dialogues: Number of dialogues to generate
        
    Returns:
        List of dialogue strings (flattened)
    """
    dialogues = []
    
    # Common patterns in restaurant bookings
    greetings = [
        "hi i'd like to make a reservation",
        "hello can i book a table",
        "good evening i need a reservation",
        "hi i want to reserve a table",
        "hello could i make a booking",
    ]
    
    time_questions = [
        "what time would you like",
        "what time are you looking for",
        "when would you like to come",
        "what time works for you",
        "what time do you prefer",
    ]
    
    party_questions = [
        "how many people",
        "table for how many",
        "how many guests",
        "party size",
        "number of people",
    ]
    
    confirmations = [
        "your reservation is confirmed",
        "your table is booked",
        "reservation confirmed",
        "i've booked your table",
        "your booking is set",
    ]
    
    times = ["6pm", "7pm", "7:30pm", "8pm", "8:30pm", "9pm", "6:30pm"]
    party_sizes = ["two", "three", "four", "five", "six", "two people", "three people", "four people"]
    names = ["john", "sarah", "mike", "emily", "david", "lisa", "chris", "anna"]
    
    for _ in range(n_dialogues):
        # Build a natural dialogue
        dialogue_parts = []
        
        # Customer greeting
        dialogue_parts.append(f"customer: {random.choice(greetings)}")
        
        # Agent response
        dialogue_parts.append("agent: sure")
        dialogue_parts.append(f"agent: {random.choice(time_questions)}")
        
        # Customer time
        time = random.choice(times)
        dialogue_parts.append(f"customer: {time}")
        
        # Agent party question
        dialogue_parts.append(f"agent: {random.choice(party_questions)}")
        
        # Customer party size
        party = random.choice(party_sizes)
        dialogue_parts.append(f"customer: {party}")
        
        # Agent confirmation
        name = random.choice(names)
        dialogue_parts.append(f"agent: {random.choice(confirmations)}")
        dialogue_parts.append(f"agent: see you at {time} {name}")
        
        # Flatten dialogue
        dialogue = " ".join(dialogue_parts)
        dialogues.append(dialogue)
    
    return dialogues


def generate_extended_dialogues(n_dialogues: int = 300) -> List[str]:
    """
    Generate more varied restaurant booking dialogues with additional patterns.
    """
    dialogues = []
    
    # Extended patterns
    special_requests = [
        "do you have outdoor seating",
        "is there parking available",
        "can we have a quiet table",
        "do you accommodate dietary restrictions",
        "is the restaurant wheelchair accessible",
    ]
    
    availability_responses = [
        "yes we have availability",
        "that time works",
        "i can accommodate that",
        "sure that's available",
        "yes we have a table",
    ]
    
    unavailable_responses = [
        "sorry that time is booked",
        "unfortunately we're full",
        "that slot isn't available",
        "we're booked at that time",
        "could you try another time",
    ]
    
    alternatives = [
        "how about 7:30pm",
        "we have 8pm available",
        "6:30pm works if that helps",
        "9pm is open",
    ]
    
    times = ["6pm", "7pm", "7:30pm", "8pm", "8:30pm", "9pm", "6:30pm", "5:30pm", "9:30pm"]
    party_sizes = ["one", "two", "three", "four", "five", "six", "two people", "three people", "four people", "a party of two", "a party of four"]
    
    for _ in range(n_dialogues):
        dialogue_parts = []
        
        # Variant 1: Standard booking
        if random.random() < 0.6:
            dialogue_parts.append("customer: hi i'd like to make a reservation")
            dialogue_parts.append("agent: sure what time")
            dialogue_parts.append(f"customer: {random.choice(times)}")
            dialogue_parts.append("agent: how many people")
            dialogue_parts.append(f"customer: {random.choice(party_sizes)}")
            dialogue_parts.append("agent: your reservation is confirmed")
        
        # Variant 2: With special request
        elif random.random() < 0.8:
            dialogue_parts.append("customer: hello can i book a table")
            dialogue_parts.append("agent: of course what time")
            dialogue_parts.append(f"customer: {random.choice(times)}")
            dialogue_parts.append("agent: how many guests")
            dialogue_parts.append(f"customer: {random.choice(party_sizes)}")
            dialogue_parts.append(f"customer: {random.choice(special_requests)}")
            dialogue_parts.append(f"agent: {random.choice(availability_responses)}")
            dialogue_parts.append("agent: your table is booked")
        
        # Variant 3: Unavailable then alternative
        else:
            dialogue_parts.append("customer: hi i need a reservation")
            dialogue_parts.append("agent: what time")
            time = random.choice(times)
            dialogue_parts.append(f"customer: {time}")
            dialogue_parts.append(f"agent: {random.choice(unavailable_responses)}")
            dialogue_parts.append(f"agent: {random.choice(alternatives)}")
            dialogue_parts.append("customer: that works")
            dialogue_parts.append("agent: how many people")
            dialogue_parts.append(f"customer: {random.choice(party_sizes)}")
            dialogue_parts.append("agent: reservation confirmed")
        
        dialogue = " ".join(dialogue_parts)
        dialogues.append(dialogue)
    
    return dialogues


def format_dialogue_for_tlm(dialogue: str) -> str:
    """
    Format a dialogue string for TLM training.
    Converts to lowercase, handles punctuation, ensures valid characters.
    
    Args:
        dialogue: Raw dialogue string
        
    Returns:
        Formatted string ready for TLM
    """
    # Convert to lowercase
    formatted = dialogue.lower()
    
    # Ensure only valid characters (replace invalid with space)
    formatted = "".join([c if c in ALPHABET else " " for c in formatted])
    
    # Collapse multiple spaces
    formatted = " ".join(formatted.split())
    
    return formatted


def collect_training_data(n_dialogues: int = 500) -> List[str]:
    """
    Collect and format restaurant booking dialogues for TLM training.
    
    Args:
        n_dialogues: Total number of dialogues to generate
        
    Returns:
        List of formatted dialogue strings
    """
    print(f"Generating {n_dialogues} restaurant booking dialogues...")
    
    # Generate base dialogues
    base_dialogues = generate_restaurant_dialogues(n_dialogues // 2)
    
    # Generate extended dialogues
    extended_dialogues = generate_extended_dialogues(n_dialogues - len(base_dialogues))
    
    # Combine and format
    all_dialogues = base_dialogues + extended_dialogues
    formatted_dialogues = [format_dialogue_for_tlm(d) for d in all_dialogues]
    
    print(f"Generated {len(formatted_dialogues)} formatted dialogues")
    print(f"Average length: {sum(len(d) for d in formatted_dialogues) / len(formatted_dialogues):.1f} characters")
    
    return formatted_dialogues


def save_dialogues(dialogues: List[str], filepath: str = "anomaly_radar/dialogues.txt"):
    """
    Save dialogues to a text file.
    
    Args:
        dialogues: List of dialogue strings
        filepath: Path to save file
    """
    import os
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    
    with open(filepath, "w") as f:
        for dialogue in dialogues:
            f.write(dialogue + "\n")
    
    print(f"Saved {len(dialogues)} dialogues to {filepath}")


def load_dialogues(filepath: str = "anomaly_radar/dialogues.txt") -> List[str]:
    """
    Load dialogues from a text file.
    
    Args:
        filepath: Path to dialogue file
        
    Returns:
        List of dialogue strings
    """
    with open(filepath, "r") as f:
        dialogues = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(dialogues)} dialogues from {filepath}")
    return dialogues


def load_real_dialogues(filepath: str) -> List[str]:
    """
    Load real dialogues from a text file and clean them for TLM training.
    
    This function:
    - Reads dialogues from file (one per line or multi-line)
    - Converts to lowercase
    - Filters to only valid alphabet characters
    - Formats for TLM training
    
    Args:
        filepath: Path to dialogue file
        
    Returns:
        List of cleaned dialogue strings
    """
    print(f"Loading real dialogues from {filepath}...")
    
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    cleaned = []
    current_dialogue = []
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines (they separate dialogues)
        if not line:
            if current_dialogue:
                # Join current dialogue and add
                dialogue = " ".join(current_dialogue)
                cleaned_dialogue = format_dialogue_for_tlm(dialogue)
                if cleaned_dialogue:  # Only add non-empty
                    cleaned.append(cleaned_dialogue)
                current_dialogue = []
            continue
        
        # Process line
        line_lower = line.lower()
        # Filter to valid characters
        line_cleaned = "".join([c if c in ALPHABET else " " for c in line_lower])
        # Collapse spaces
        line_cleaned = " ".join(line_cleaned.split())
        
        if line_cleaned:
            current_dialogue.append(line_cleaned)
    
    # Add last dialogue if exists
    if current_dialogue:
        dialogue = " ".join(current_dialogue)
        cleaned_dialogue = format_dialogue_for_tlm(dialogue)
        if cleaned_dialogue:
            cleaned.append(cleaned_dialogue)
    
    print(f"Loaded and cleaned {len(cleaned)} real dialogues")
    if cleaned:
        avg_length = sum(len(d) for d in cleaned) / len(cleaned)
        print(f"Average length: {avg_length:.1f} characters")
        print(f"Sample: {cleaned[0][:80]}...")
    
    return cleaned


def load_real_dialogues_simple(filepath: str) -> List[str]:
    """
    Simple version: one dialogue per line.
    
    Args:
        filepath: Path to dialogue file (one dialogue per line)
        
    Returns:
        List of cleaned dialogue strings
    """
    print(f"Loading real dialogues from {filepath}...")
    
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Clean and format
        cleaned_dialogue = format_dialogue_for_tlm(line)
        if cleaned_dialogue:
            cleaned.append(cleaned_dialogue)
    
    print(f"Loaded and cleaned {len(cleaned)} real dialogues")
    if cleaned:
        avg_length = sum(len(d) for d in cleaned) / len(cleaned)
        print(f"Average length: {avg_length:.1f} characters")
        print(f"Sample: {cleaned[0][:80]}...")
    
    return cleaned


if __name__ == "__main__":
    # Generate and save dialogues
    print("=" * 60)
    print("Restaurant Booking Dialogue Data Collection")
    print("=" * 60)
    
    dialogues = collect_training_data(n_dialogues=500)
    
    # Show samples
    print("\nSample dialogues:")
    for i, dialogue in enumerate(dialogues[:5], 1):
        print(f"\n{i}. {dialogue[:100]}...")
    
    # Save
    save_dialogues(dialogues, "anomaly_radar/dialogues.txt")
    
    print("\n" + "=" * 60)
    print("Data collection complete!")
    print("=" * 60)

