#!/usr/bin/env python3
"""
Task 10: Implement Text Processor
Extracts early clinical notes from discharge.csv, applies temporal
filters, and generates BioClinicalBERT embeddings for the AKI Prediction Pipeline.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import torch
from transformers import AutoTokenizer, AutoModel
from datetime import timedelta
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class TextProcessor:
    def __init__(self):
        self.logger = logging.getLogger('text_processor')
        self.setup_logging()
        
        self.raw_data_dir = config.RAW_DATA_DIR
        self.processed_dir = config.PROCESSED_DATA_DIR
        
        # Load BioClinicalBERT
        self.model_name = "emilyalsentzer/Bio_ClinicalBERT"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def setup_logging(self):
        if not logging.getLogger('text_processor').handlers:
            logging.basicConfig(
                level=logging.INFO,
                format=config.LOG_FORMAT,
                handlers=[logging.StreamHandler(sys.stdout)]
            )

    def load_cohort(self) -> pd.DataFrame:
        self.logger.info("Loading labeled cohorts...")
        cohort_path = os.path.join(self.processed_dir, 'labeled_stays.csv')
        stays_df = pd.read_csv(cohort_path)
        stays_df['intime'] = pd.to_datetime(stays_df['intime'])
        return stays_df

    def clean_text(self, text: str) -> str:
        """Basic clinical text cleaning."""
        if not isinstance(text, str):
            return ""
        # Remove multiple newlines and spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove de-id tags like [**...**]
        text = re.sub(r'\[\*\*[^\]]+\*\*\]', '', text)
        return text.strip().lower()

    def process_notes(self):
        stays_df = self.load_cohort()
        valid_hadm_ids = set(stays_df['hadm_id'].dropna().unique())
        
        notes_path = os.path.join(self.raw_data_dir, 'discharge.csv')
        
        self.logger.info("Loading discharge.csv in chunks...")
        chunks = []
        total_rows = 0
        for chunk in pd.read_csv(notes_path, chunksize=100_000, usecols=['hadm_id', 'charttime', 'text']):
            total_rows += len(chunk)
            # Filter by admission IDs in cohort
            chunk = chunk[chunk['hadm_id'].isin(valid_hadm_ids)].copy()
            if len(chunk) > 0:
                chunks.append(chunk)
                
        if not chunks:
            self.logger.warning("No notes found for the selected cohort!")
            return
            
        notes_df = pd.concat(chunks, ignore_index=True)
        notes_df['charttime'] = pd.to_datetime(notes_df['charttime'], errors='coerce')
        self.logger.info(f"Loaded {len(notes_df)} total notes for cohort.")
        
        # Report text coverage
        stays_with_text = notes_df['hadm_id'].nunique()
        total_stays = len(valid_hadm_ids)
        self.logger.info(f"ICU stays with discharge summaries: {stays_with_text}/{total_stays} ({100*stays_with_text/total_stays:.1f}%)")
        
        # Merge with stays to get intime and stay_id
        stay_mapping = stays_df[['hadm_id', 'stay_id', 'intime']]
        # One hadm_id can have multiple notes and multiple stays! 
        merged = notes_df.merge(stay_mapping, on='hadm_id', how='inner')
        
        # TEMPORAL FILTERING: Keep notes written <= intime + 24h
        self.logger.info("Applying < 24h temporal filtering to prevent data leakage...")
        cutoff_time = merged['intime'] + pd.Timedelta(hours=config.TEMPORAL_CUTOFF_HOURS)
        early_notes = merged[
            (merged['charttime'].isna()) |  # If no charttime, assume it might be valid or drop? Let's drop.
            (merged['charttime'] <= cutoff_time)
        ].copy()
        
        early_notes.dropna(subset=['charttime'], inplace=True)
        self.logger.info(f"Retained {len(early_notes)} notes within the 24h window.")
        
        if len(early_notes) > 0:
            # Clean text
            self.logger.info("Cleaning text...")
            early_notes['cleaned_text'] = early_notes['text'].apply(self.clean_text)
            
            # Aggregate per stay
            self.logger.info("Aggregating text per stay_id...")
            stay_text = early_notes.groupby('stay_id')['cleaned_text'].apply(lambda x: " ".join(x)).reset_index()
        else:
            stay_text = pd.DataFrame(columns=['stay_id', 'cleaned_text'])

        # Now, generate embeddings for ALL stays in stays_df, even those without notes (zero vector)
        self.logger.info(f"Loading {self.model_name} on {self.device}...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name).to(self.device)
        model.eval()

        embed_dim = 768
        
        all_stay_ids = list(stays_df['stay_id'].unique())
        text_dict = dict(zip(stay_text['stay_id'], stay_text['cleaned_text']))
        
        embeddings = []
        
        self.logger.info("Generating embeddings (Batch tokenization) ...")
        
        # We will process in batches to save memory
        batch_size = 64
        empty_vec = np.zeros(embed_dim, dtype=np.float32)
        
        for i in range(0, len(all_stay_ids), batch_size):
            batch_ids = all_stay_ids[i : i + batch_size]
            batch_texts = [text_dict.get(sid, "") for sid in batch_ids]
            
            # Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                # Use [CLS] token representation (pooler_output or index 0 of last_hidden_state)
                # Standard BERT uses pooler_output
                cls_embeds = outputs.pooler_output.cpu().numpy()
                
            # For empty texts, manually zero out
            for j, txt in enumerate(batch_texts):
                if len(txt.strip()) == 0:
                    cls_embeds[j] = empty_vec
                    
            embeddings.append(cls_embeds)
            
            # Checkpoint every 1000 samples
            processed = i + len(batch_ids)
            if processed % config.CHECKPOINT_INTERVAL < batch_size:
                checkpoint_path = os.path.join(self.processed_dir, 'embedding_checkpoint.npy')
                checkpoint_ids_path = os.path.join(self.processed_dir, 'embedding_checkpoint_ids.npy')
                np.save(checkpoint_path, np.vstack(embeddings))
                np.save(checkpoint_ids_path, np.array(all_stay_ids[:processed]))
                self.logger.info(f"  Checkpoint saved at {processed}/{len(all_stay_ids)} stays")
            
            if (i + batch_size) % 10000 < batch_size:
                self.logger.info(f"  Embedded {i + batch_size}/{len(all_stay_ids)} stays...")
                
        all_embeddings = np.vstack(embeddings)
        self.logger.info(f"Generated embeddings shape: {all_embeddings.shape}")
        
        # Save to disk
        out_path = os.path.join(self.processed_dir, 'text_embeddings.npy')
        ids_path = os.path.join(self.processed_dir, 'text_stay_ids.npy')
        
        np.save(out_path, all_embeddings)
        np.save(ids_path, np.array(all_stay_ids))
        
        self.logger.info(f"Saved text embeddings to {out_path}")
        self.logger.info(f"Saved stay IDs to {ids_path}")
        
        # Remove checkpoint files if they exist
        for cp in ['embedding_checkpoint.npy', 'embedding_checkpoint_ids.npy']:
            cp_path = os.path.join(self.processed_dir, cp)
            if os.path.exists(cp_path):
                os.remove(cp_path)
                self.logger.info(f"Removed temporary checkpoint: {cp}")
        
        # Save split-level text embeddings
        self.logger.info("Saving split-level text embeddings...")
        id_to_idx = {sid: i for i, sid in enumerate(all_stay_ids)}
        
        for split_name in ['train', 'val', 'test']:
            pts_path = os.path.join(self.processed_dir, f'{split_name}_patients.csv')
            if not os.path.exists(pts_path):
                self.logger.warning(f"Cannot find {pts_path} - skipping split text embedding save.")
                continue
            pts = set(pd.read_csv(pts_path)['subject_id'])
            struct_path = os.path.join(self.processed_dir, 'structured_dataset.csv')
            df_struct = pd.read_csv(struct_path, usecols=['subject_id', 'stay_id'])
            split_sids = df_struct[df_struct['subject_id'].isin(pts)]['stay_id'].values
            
            split_embeds = []
            for sid in split_sids:
                idx = id_to_idx.get(sid)
                if idx is not None:
                    split_embeds.append(all_embeddings[idx])
                else:
                    split_embeds.append(np.zeros(embed_dim, dtype=np.float32))
            X_split = np.vstack(split_embeds)
            split_path = os.path.join(self.processed_dir, f'X_{split_name}_text.npy')
            np.save(split_path, X_split)
            self.logger.info(f"  Saved {split_path} shape: {X_split.shape}")
        
        # Document temporal limitations
        self.logger.info("TEMPORAL LIMITATION: Discharge summaries may contain post-ICU information.")
        self.logger.info("This is a known limitation for representation analysis.")
        self.logger.info("Future work should use temporally aligned clinical notes (e.g., nursing notes within 24h).")

def main():
    processor = TextProcessor()
    processor.process_notes()

if __name__ == '__main__':
    main()
