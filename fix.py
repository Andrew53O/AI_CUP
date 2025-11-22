#!/usr/bin/env python3

def process_flag(flag: str) -> str:
    # Ensure even length
    if len(flag) % 2 != 0:
        raise ValueError("Input length must be even (pairs of characters).")

    # Process into 16-bit characters
    result = ''.join(
        chr((ord(flag[i]) << 8) + ord(flag[i + 1]))
        for i in range(0, len(flag), 2)
    )
    return result


def main():
    input_file = "input.txt"      # change if needed
    output_file = "output.txt"    # change if needed

    # Read the file
    with open(input_file, "r", encoding="utf-8") as f:
        data = f.read().strip()

    # Process
    processed = process_flag(data)

    # Save output
    with open(output_file, "w", encoding="utf-16") as f:
        f.write(processed)

    print(f"[✔] Processed data written to {output_file}")


if __name__ == "__main__":
    main()
