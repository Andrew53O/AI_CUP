# clean_predictions.py

def keep_highest_conf(input_file, output_file):
    best = {}  # {filename: (cls, conf, x1, y1, x2, y2)}

    with open(input_file, "r") as fin:
        for line in fin:
            parts = line.strip().split()
            if len(parts) != 7:
                continue  # skip broken lines

            filename = parts[0]
            cls = int(parts[1])
            conf = float(parts[2])
            x1, y1, x2, y2 = map(int, parts[3:7])

            # keep only highest confidence
            if filename not in best or conf > best[filename][1]:
                best[filename] = (cls, conf, x1, y1, x2, y2)

    # write cleaned output
    with open(output_file, "w") as fout:
        for filename in sorted(best.keys()):
            cls, conf, x1, y1, x2, y2 = best[filename]
            fout.write(f"{filename} {cls} {conf:.4f} {x1} {y1} {x2} {y2}\n")

    print(f"Cleaned file saved to: {output_file}")


if __name__ == "__main__":
    input_path = "merged.txt"          # CHANGE IF NEEDED
    output_path = "cleaned_prediction.txt" # output filename

    keep_highest_conf(input_path, output_path)
