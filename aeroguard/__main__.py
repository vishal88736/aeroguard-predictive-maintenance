"""Run AeroGuard locally: .venv/bin/python -m aeroguard."""

import argparse
import logging
import os

# Prevent implicit external telemetry and bound CPU work before numerical imports.
os.environ.setdefault('MLFLOW_DISABLE_TELEMETRY','true')
os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')

from fleetpm.config import ROOT


def main():
    import uvicorn
    from .api import create_app
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default=os.environ.get('AEROGUARD_HOST','127.0.0.1'))
    parser.add_argument('--port',type=int,default=int(os.environ.get('AEROGUARD_PORT','8765')))
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(name)s %(message)s')
    uvicorn.run(create_app(),host=args.host,port=args.port,log_level='info')


if __name__=='__main__':main()
