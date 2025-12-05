"""
Generate Anomalous Dataset for Benchmarking

Creates ~200 examples of various anomaly types:
- Hallucinated outputs
- Off-topic generations
- Repetitive patterns
- Broken syntax
- Mixed-domain weirdness
- Numeric junk / symbol spam
"""

import random
from typing import List


def generate_hallucinated_outputs(n: int = 40) -> List[str]:
    """Generate hallucinated/nonsensical restaurant booking outputs."""
    anomalies = []
    
    patterns = [
        "quantum computing research table booking for negative infinity people at yesterday time",
        "i need a reservation for my pet dragon at 25pm in the fourth dimension",
        "book a table for zero people at imaginary time for quantum dinner",
        "reservation for two at 7pm confirmed your table is in another universe",
        "i want to book a table made of light for seven negative people",
        "can i reserve a table for yesterday at tomorrow time please",
        "table booking for infinity guests at quantum hour confirmed",
        "reservation for my imaginary friends at nonexistent time",
        "book a table for negative five people at 99pm yesterday",
        "i need a reservation in the past for future people",
        "table for two at 7pm but also for quantum particles",
        "reservation confirmed for table in parallel universe",
        "book dinner for my thoughts at thought time",
        "i want a table for the number pi people at e oclock",
        "reservation for abstract concepts at conceptual time",
    ]
    
    # Generate variations
    for _ in range(n):
        base = random.choice(patterns)
        # Add random variations
        variations = [
            base,
            base + " please confirm",
            base + " thank you",
            "hello " + base,
            base.replace("table", "quantum table"),
            base.replace("people", "entities"),
        ]
        anomalies.append(random.choice(variations))
    
    return anomalies


def generate_off_topic_outputs(n: int = 40) -> List[str]:
    """Generate off-topic content that's not restaurant-related."""
    anomalies = []
    
    topics = [
        "the sky is blue and elephants can fly through quantum space",
        "i am writing a novel about artificial intelligence and machine learning",
        "the weather today is sunny with a chance of rain tomorrow",
        "mathematics is the language of the universe and quantum mechanics",
        "i love programming in python and building neural networks",
        "the stock market is volatile and cryptocurrency is unpredictable",
        "cooking recipes require precise measurements and timing",
        "travel destinations around the world offer unique experiences",
        "music theory involves harmony rhythm and melody composition",
        "sports require physical fitness and strategic thinking",
        "philosophy explores the nature of existence and consciousness",
        "history teaches us about past civilizations and cultures",
        "science fiction explores future technologies and possibilities",
        "art expresses emotions through visual and creative mediums",
        "literature tells stories that connect us across time",
    ]
    
    # Mix with restaurant keywords to make it subtle
    restaurant_words = ["table", "reservation", "booking", "dinner", "restaurant"]
    
    for _ in range(n):
        topic = random.choice(topics)
        if random.random() < 0.5:
            # Pure off-topic
            anomalies.append(topic)
        else:
            # Mixed (subtle anomaly)
            word = random.choice(restaurant_words)
            anomalies.append(f"{topic} {word} {random.choice(restaurant_words)}")
    
    return anomalies


def generate_repetitive_outputs(n: int = 30) -> List[str]:
    """Generate repetitive/stuck outputs."""
    anomalies = []
    
    patterns = [
        "hello hello hello hello hello hello hello hello",
        "yes yes yes yes yes yes yes yes",
        "ok ok ok ok ok ok ok ok",
        "sure sure sure sure sure sure sure sure",
        "table table table table table table",
        "reservation reservation reservation reservation",
        "confirmed confirmed confirmed confirmed confirmed",
        "7pm 7pm 7pm 7pm 7pm 7pm 7pm",
        "two two two two two two two",
        "thank you thank you thank you thank you",
    ]
    
    for _ in range(n):
        base = random.choice(patterns)
        # Vary length
        repeats = random.randint(5, 15)
        word = base.split()[0]
        anomalies.append(" ".join([word] * repeats))
    
    return anomalies


def generate_broken_syntax(n: int = 30) -> List[str]:
    """Generate broken syntax/grammar."""
    anomalies = []
    
    patterns = [
        "table for two at 7pm reservation booking confirmed your",
        "i want reservation table two people 7pm time",
        "book table two 7pm confirm reservation",
        "reservation table two 7pm confirmed booking",
        "table booking two people 7pm reservation confirm",
        "i need table reservation two 7pm booking",
        "book table two 7pm reservation confirm",
        "reservation table two 7pm booking confirm",
        "table two people 7pm reservation booking",
        "i want table reservation two 7pm booking confirm",
    ]
    
    # Add random word order variations
    for _ in range(n):
        base = random.choice(patterns)
        words = base.split()
        if random.random() < 0.3:
            # Shuffle some words
            random.shuffle(words)
            anomalies.append(" ".join(words))
        else:
            anomalies.append(base)
    
    return anomalies


def generate_mixed_domain(n: int = 30) -> List[str]:
    """Generate mixed-domain weirdness."""
    anomalies = []
    
    domains = [
        ("restaurant", "quantum physics"),
        ("booking", "machine learning"),
        ("table", "neural networks"),
        ("reservation", "astronomy"),
        ("dinner", "programming"),
        ("restaurant", "mathematics"),
        ("booking", "philosophy"),
    ]
    
    for _ in range(n):
        domain1, domain2 = random.choice(domains)
        anomalies.append(
            f"i want to {domain1} a {random.choice(['table', 'reservation', 'booking'])} "
            f"for {random.choice(['two', 'three', 'four'])} at 7pm using {domain2} principles"
        )
    
    return anomalies


def generate_numeric_junk(n: int = 30) -> List[str]:
    """Generate numeric junk and symbol spam."""
    anomalies = []
    
    patterns = [
        "table for 999999 people at 99pm",
        "reservation for -5 people at 25pm",
        "booking for 0 people at 0pm",
        "table for 1000000 guests at 100pm",
        "reservation for 2.71828 people at 3.14159pm",
        "booking for infinity people at negative time",
        "table for 123456789 people at 987654321pm",
        "reservation for ### people at $$$pm",
        "booking for @@@ people at !!!pm",
        "table for %%% people at &&&pm",
    ]
    
    for _ in range(n):
        base = random.choice(patterns)
        # Add more junk
        if random.random() < 0.5:
            base += " " + "".join([random.choice(["#", "@", "!", "%", "$"])] * random.randint(3, 10))
        anomalies.append(base)
    
    return anomalies


def generate_all_anomalies(n_total: int = 200) -> List[tuple]:
    """
    Generate all types of anomalies.
    
    Returns:
        List of (text, anomaly_type) tuples
    """
    all_anomalies = []
    
    # Distribute across types
    n_per_type = n_total // 6
    
    all_anomalies.extend([(text, "hallucinated") for text in generate_hallucinated_outputs(n_per_type)])
    all_anomalies.extend([(text, "off_topic") for text in generate_off_topic_outputs(n_per_type)])
    all_anomalies.extend([(text, "repetitive") for text in generate_repetitive_outputs(n_per_type // 2)])
    all_anomalies.extend([(text, "broken_syntax") for text in generate_broken_syntax(n_per_type // 2)])
    all_anomalies.extend([(text, "mixed_domain") for text in generate_mixed_domain(n_per_type // 2)])
    all_anomalies.extend([(text, "numeric_junk") for text in generate_numeric_junk(n_per_type // 2)])
    
    # Shuffle
    random.shuffle(all_anomalies)
    
    return all_anomalies[:n_total]


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from anomaly_radar.dialogue_data import format_dialogue_for_tlm
    
    print("=" * 70)
    print("Generating Anomalous Dataset for Benchmarking")
    print("=" * 70)
    
    anomalies = generate_all_anomalies(n_total=200)
    
    # Format and save
    formatted = [(format_dialogue_for_tlm(text), anomaly_type) for text, anomaly_type in anomalies]
    
    # Save to file
    output_file = "anomaly_radar/anomalous_dataset.txt"
    with open(output_file, "w") as f:
        for text, anomaly_type in formatted:
            f.write(f"{text}\t{anomaly_type}\n")
    
    print(f"\n✓ Generated {len(formatted)} anomalous examples")
    print(f"✓ Saved to {output_file}")
    
    # Show samples
    print("\nSample anomalies:")
    for i, (text, anomaly_type) in enumerate(formatted[:5], 1):
        print(f"\n{i}. [{anomaly_type}] {text[:70]}...")
    
    print("\n" + "=" * 70)
    print("Anomaly dataset generation complete!")
    print("=" * 70)

