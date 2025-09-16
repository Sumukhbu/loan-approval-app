
# Simple script to generate or update model card from metadata
import json, os
def update_model_card(artifact_dir='artifacts'):
    meta_path = os.path.join(artifact_dir,'metadata.json')
    if not os.path.exists(meta_path):
        print('No metadata found at', meta_path)
        return
    meta = json.load(open(meta_path))
    with open(os.path.join(artifact_dir,'MODEL_CARD_SUMMARY.txt'),'w') as f:
        f.write('Model metadata summary:\n')
        f.write(json.dumps(meta, indent=2))
        print('Wrote model card summary')
if __name__=='__main__':
    update_model_card()
