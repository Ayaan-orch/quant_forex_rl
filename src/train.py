import argparse
import shutil
import sys
from pathlib import Path
import pandas as pd
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

# Ensure src is in sys.path
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from environment import AdvancedForexEnv

DATA_PATH = ROOT_DIR / "data" / "processed_market_features.csv"
MODEL_PATH = ROOT_DIR / "ppo_forex_policy"
BACKUP_PATH = ROOT_DIR / "ppo_forex_policy_prev.zip"

def run_training(total_timesteps: int = 50_000, use_lstm: bool = True, resume: bool = False):
    print(f"\n[Training Pipeline] Loading market features from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # 80% chronological train partition
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)

    print(f"Training on pullbacks in {len(train_df):,} bars (80% train partition of {len(df):,} total bars)...")
    train_env = AdvancedForexEnv(train_df, random_start=True)

    saved_zip = MODEL_PATH.with_suffix(".zip")
    if resume and saved_zip.exists():
        print(f"\n[Resume Mode] Loading existing weights from {saved_zip}...")
        try:
            model = RecurrentPPO.load(str(MODEL_PATH), env=train_env)
            print(f"Successfully loaded RecurrentPPO policy (already trained for {model.num_timesteps:,} steps).")
        except Exception:
            model = PPO.load(str(MODEL_PATH), env=train_env)
            print(f"Successfully loaded standard PPO policy (already trained for {model.num_timesteps:,} steps).")
        
        print(f"Continuing training for an additional {total_timesteps:,} steps...")
        reset_timesteps = False
    else:
        if resume and not saved_zip.exists():
            print(f"\n[Notice] No existing model found at {saved_zip}. Starting fresh training run...")
        
        reset_timesteps = True
        if use_lstm:
            print("\nConfiguring fresh Recurrent PPO with LSTM memory policy (MlpLstmPolicy)...")
            model = RecurrentPPO(
                policy="MlpLstmPolicy",
                env=train_env,
                learning_rate=3e-4,
                n_steps=256,
                batch_size=64,
                n_epochs=5,
                gamma=0.95,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.06,
                policy_kwargs=dict(
                    lstm_hidden_size=64,
                    n_lstm_layers=1,
                    net_arch=dict(pi=[64], vf=[64])
                ),
                verbose=1,
                seed=42
            )
        else:
            print("\nConfiguring standard Feedforward PPO (MlpPolicy)...")
            model = PPO(
                policy="MlpPolicy",
                env=train_env,
                learning_rate=3e-4,
                n_steps=512,
                batch_size=64,
                n_epochs=10,
                gamma=0.95,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.015,
                verbose=1,
                seed=42
            )
        print(f"\nInitiating fresh agent training for {total_timesteps:,} steps...")

    # Backup existing checkpoint before overwriting
    if saved_zip.exists():
        shutil.copyfile(saved_zip, BACKUP_PATH)
        print(f"Saved backup of previous policy to '{BACKUP_PATH}'.")

    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=reset_timesteps)

    model.save(str(MODEL_PATH))
    print(f"\nTrained policy successfully serialized to '{MODEL_PATH}.zip'.")
    print(f"Total cumulative steps trained: {model.num_timesteps:,}")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or continue training the Quant Forex RL policy.")
    parser.add_argument("--timesteps", type=int, default=50_000, help="Number of timesteps to train (default: 50000)")
    parser.add_argument("--resume", action="store_true", help="Continue training from the existing ppo_forex_policy.zip")
    parser.add_argument("--no-lstm", action="store_true", help="Use standard feedforward PPO instead of RecurrentPPO")
    
    args = parser.parse_args()
    run_training(
        total_timesteps=args.timesteps,
        use_lstm=not args.no_lstm,
        resume=args.resume
    )