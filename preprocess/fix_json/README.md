# Fix faulty json files

This is to fix faulty files, particularly more than two objects are on the same line. 

## Validate and get the faulty file list
```bash
ls *.jsonl.gz > files.txt
python validate_json.py $(cat files.txt) --output fault.txt
```

## Fix the files
