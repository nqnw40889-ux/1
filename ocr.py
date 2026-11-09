"""Read the six labeled fields on a sensor report image."""
import io
import re
import shutil
import subprocess

from PIL import Image, ImageOps

from core import FEATURES

IMAGE_LABELS = ['TYPE', 'AIR K', 'PROCESS K', 'SPEED RPM', 'TORQUE NM', 'WEAR MIN']


def read_sensor_image(payload):
    if len(payload) > 5 * 1024 * 1024:
        raise ValueError('Ảnh vượt 5 MB.')
    engine = shutil.which('tesseract')
    if engine is None:
        raise ValueError('Cần cài Tesseract OCR và thêm vào PATH để đọc ảnh.')
    with Image.open(io.BytesIO(payload)) as raw:
        if raw.format not in ('PNG', 'JPEG') or raw.width * raw.height > 12_000_000:
            raise ValueError('Chọn ảnh JPG hoặc PNG tối đa 12 megapixel.')
        image = ImageOps.autocontrast(ImageOps.grayscale(ImageOps.exif_transpose(raw)))
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    try:
        process = subprocess.run(
            [engine, 'stdin', 'stdout', '-l', 'eng', '--psm', '6'],
            input=buffer.getvalue(), capture_output=True, timeout=12, check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ValueError('Chưa đọc được bảng thông số trong ảnh.') from exc

    fields = {}
    text = process.stdout.decode('utf-8', errors='replace').upper()
    for label, feature in zip(IMAGE_LABELS, FEATURES):
        value_pattern = r'[LMH]' if feature == 'Type' else r'[-+]?\d+(?:[.,]\d+)?'
        matches = re.findall(rf'^\s*{label}\s*[:=]?\s*({value_pattern})\s*$', text, flags=re.M)
        fields[feature] = matches[0].replace(',', '.') if len(matches) == 1 else ''
    return fields
