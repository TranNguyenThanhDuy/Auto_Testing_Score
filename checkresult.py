import json
import csv

# Duong dan toi file JSON
ANSWERS_FILE = 'answers.json'
RESULT_FILE   = 'result3.json'
OUTPUT_CSV    = 'grading_result.csv'

# Bien luu so ky tu mong doi cho sbd va ma_de (se gan trong main)
EXPECTED_SBD_LEN  = None
EXPECTED_MADE_LEN = None

def decode_bubbles(bubbles_dict, label):
    decoded = ''
    errors = []
    for pos in sorted(bubbles_dict, key=lambda x: int(x)):
        digit_map = bubbles_dict[pos]
        filled = [d for d, v in digit_map.items() if v == 1]
        if len(filled) == 1:
            decoded += filled[0]
        elif len(filled) == 0:
            errors.append(f"Khong to {label} ky tu thu {pos}")
        else:
            errs = ','.join(filled)
            errors.append(f"To nhieu o o {label} ky tu thu {pos}: {errs}")
    return decoded, errors


def grade_one(result_data, answer_key):
    marks = {}
    errors = []
    for q in range(1, 41):
        q_str = str(q)
        bubbles = result_data.get('answers', {}).get(q_str, {})
        filled = [opt for opt, v in bubbles.items() if v == 1]
        if len(filled) == 0:
            chosen = None
            errors.append(f"Cau {q}: khong to o nao")
        elif len(filled) > 1:
            chosen = None
            errs = ','.join(filled)
            errors.append(f"Cau {q}: to nhieu o ({errs})")
        else:
            chosen = filled[0]

        correct = answer_key.get(q_str)
        marks[q_str] = 'Dung' if chosen == correct else 'Sai'
    return marks, errors


def main():
    # 1. Load du lieu
    with open(ANSWERS_FILE, 'r', encoding='utf-8') as f:
        answers_data = json.load(f)
    with open(RESULT_FILE, 'r', encoding='utf-8') as f:
        result_data = json.load(f)

    # 2. Xac dinh do dai sbd va ma_de tu file result
    global EXPECTED_SBD_LEN, EXPECTED_MADE_LEN
    EXPECTED_SBD_LEN  = len(result_data.get('sbd', {}))
    EXPECTED_MADE_LEN = len(result_data.get('ma_de', {}))

    # 3. Giai ma sbd va ma_de
    sbd, errors_sbd    = decode_bubbles(result_data.get('sbd', {}),   'sbd')
    ma_de, errors_made = decode_bubbles(result_data.get('ma_de', {}), 'ma_de')

    # 3b. Kiem tra so ky tu neu khong du
    if len(sbd) < EXPECTED_SBD_LEN:
        errors_sbd.append(
            f"sbd khong du ky tu: du kien {EXPECTED_SBD_LEN}, thuc te {len(sbd)}"
        )
    if len(ma_de) < EXPECTED_MADE_LEN:
        errors_made.append(
            f"ma_de khong du ky tu: du kien {EXPECTED_MADE_LEN}, thuc te {len(ma_de)}"
        )

    # 4. Lay dap an dung cho ma de
    if ma_de not in answers_data:
        raise ValueError(f"Khong tim thay ma de '{ma_de}' trong file answers.json")
    answer_key = answers_data[ma_de]['answers']

    # 5. Cham diem tung cau
    marks, errors_ans = grade_one(result_data, answer_key)

    # 6. Tinh so cau sai va diem
    wrong_count = sum(1 for v in marks.values() if v == 'Dung')
    diem = f"{wrong_count}|40"

    # 7. Tong hop cac loi
    all_errors = errors_sbd + errors_made + errors_ans
    notes = '; '.join(all_errors)

    # 8. Ghi CSV
    header = ['stt', 'so bao danh', 'diem', 'ma de'] + [str(i) for i in range(1, 41)] + ['ghi chu']
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        row = [1, sbd, diem, ma_de] + [marks[str(i)] for i in range(1, 41)] + [notes]
        writer.writerow(row)

    print(f"Hoan thanh cham diem, ket qua luu vao '{OUTPUT_CSV}'")

if __name__ == '__main__':
    main()
