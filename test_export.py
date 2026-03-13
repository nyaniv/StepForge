"""Quick test: export a single .pth file to verify the fix works."""
import sys, os, glob, torch

sys.path.insert(0, '/Users/nadavyaniv/step llm plan/Text2CAD/CadSeqProc')
sys.path.insert(0, '/Users/nadavyaniv/step llm plan/Text2CAD')

from cad_sequence import CADSequence

files = sorted(glob.glob('/Users/nadavyaniv/step llm plan/cad_seq/**/*.pth', recursive=True))
print(f"Found {len(files)} .pth files")

success, fail = 0, 0
for f in files[:20]:   # test first 20
    try:
        data = torch.load(f, weights_only=False)
        vec = data['vec']
        cad_seq = CADSequence.from_vec(vec['cad_vec'])
        cad_seq.create_cad_model()
        out = f'/tmp/test_{os.path.basename(f)}.step'
        cad_seq.save_stp(filename=f'test_{os.path.basename(f)}', output_folder='/tmp')
        out = f'/tmp/test_{os.path.basename(f)}.step'
        size = os.path.getsize(out)
        print(f'  OK  {os.path.basename(f)}  ({size} bytes)')
        success += 1
    except Exception as e:
        print(f'  FAIL {os.path.basename(f)}: {e}')
        fail += 1

print(f'\nResult: {success} succeeded, {fail} failed out of 20')
