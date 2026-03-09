input_file = r"D:\Financeinsight\data\clean\ner_data.bio"
output_file = r"D:\Financeinsight\data\clean\ner_data_fixed.bio"


with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

fixed = []
i = 0

while i < len(lines):
    line = lines[i].strip()

    if i + 3 < len(lines):
        t1 = lines[i].strip().split()
        t2 = lines[i+1].strip().split()
        t3 = lines[i+2].strip().split()
        t4 = lines[i+3].strip().split()

        if (
            len(t1) == 2 and len(t2) == 2 and len(t3) == 2 and len(t4) == 2 and
            t1[1] == "B-PERCENT" and
            t2[1] == "I-PERCENT" and
            t3[1] == "O" and
            t4[1] == "B-PERCENT"
        ):
            fixed.append(f"{t1[0]}\tB-PERCENT\n")
            fixed.append(f"{t2[0]}\tI-PERCENT\n")
            fixed.append(f"{t3[0]}\tI-PERCENT\n")
            fixed.append(f"{t4[0]}\tI-PERCENT\n")
            i += 4
            continue

    fixed.append(lines[i])
    i += 1

with open(output_file, "w", encoding="utf-8") as f:
    f.writelines(fixed)

print("✅ Percent-range BIO errors fixed!")
