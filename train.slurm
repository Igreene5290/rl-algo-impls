#!/bin/sh
#SBATCH --gpus-per-node=1
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

cd ~/rl-algo-impls

source .venv/bin/activate
export JAVA_HOME=/usr/lib/jvm/jre/

ENV=Microrts-env16-80m-ent5-lr3c-mgn2-shaped-rew-nga-a100

./scripts/setup.sh --microrts
./scripts/microrts.sh -a ppo -e $ENV