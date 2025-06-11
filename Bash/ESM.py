import esm
import numpy as np
import os
import pandas as pd
import random
import sys
import time
import torch



# Check for input
if len(sys.argv) < 8:
    print("ERROR: Incorrect number of inputs\n"
          "Usage: python convertToJPG.py <input file>")
    sys.exit(1)

# Get params
modelParams = sys.argv[1]
enzymeName = sys.argv[2]
fixAA = sys.argv[3]
fixPos = sys.argv[4]
lenSubs = int(sys.argv[5])
useReadingFrame = sys.argv[6]
minSubs = sys.argv[7]
batchSize = int(sys.argv[8])

# Parameters: Dataset
labelsXAxis = [f'R{i}' for i in range(1, lenSubs+1)]
enzyme = None
if enzymeName.lower() == 'mpro2':
    enzyme = f'SARS-CoV-2 M{'ᵖʳᵒ'}'
tagFile = f'{enzyme} - {fixAA}@R{fixPos}'
fileName = None
if useReadingFrame:
    fileName = f'fixedMotifSubs - {tagFile} - FinalSort - MinCounts {minSubs}'
else:
    fileName = f'fixedSubs - {tagFile} - FinalSort - MinCounts {minSubs}'


print(f'Generate Embeddings:\n'
      f'    ESM: {modelParams}\n'
      f'    Enzyme: {enzymeName}\n'
      f'    Min Subs: {minSubs}\n\n')

# Define: Directories
pathData = os.path.join('Data')
pathEmbeddings = os.path.join('Embeddings')
os.makedirs(pathData, exist_ok=True)
os.makedirs(pathEmbeddings, exist_ok=True)

# Load data
def loadSubs(file):
    print('================================= Loading Data '
          '==================================')
    pathSubs = os.path.join(pathData, fileName)
    print(f'Loading Substrates:\n'
          f'     {pathSubs}\n\n')
    return pathSubs
substrates = loadSubs(file=fileName)


substrates = {
    'AVLQSASA': 125544,
    'TSLQGVFA': 119235,
    'VILQGGTA': 110987,
}



# Set device
print('============================== Set Training Device '
      '==============================')
if torch.cuda.is_available():
    device = 'cuda:0'
    print(f'Train with Device: {device}\n'
          f'Device Name: {torch.cuda.get_device_name(device)}'
          f'\n\n')
else:
    import platform
    device = 'cpu'
    print(f'Train with Device: {device}\n'
          f'Device Name: {platform.processor()}\n\n')



def ESM(substrates, paramsESM, pathSave, trainingSet=False):
    print('=========================== Generate Embeddings: ESM '
          '============================')
    # Inspect: Data type
    predictions = True
    if trainingSet:
        predictions = False
    print(f'ESM Model: {paramsESM}\n'
          f'Batch Size: {batchSize}\n')

    # Load: ESM Embeddings
    pathEmbeddings = os.path.join(pathSave, f'{paramsESM}.csv')

    # # Generate Embeddings
    # Step 1: Convert substrates to ESM model format and generate Embeddings
    totalSubActivity = 0
    subs = []
    values = []
    if trainingSet:
        for index, (substrate, value) in enumerate(substrates.items()):
            totalSubActivity += value
            subs.append((f'Sub{index}', substrate))
            values.append(value)
    else:
        for index, substrate in enumerate(substrates):
            subs.append((f'Sub{index}', substrate))
    sampleSize = len(substrates)
    print(f'Total unique substrates: {len(substrates):,}\n'
          f'Collected substrates: {sampleSize:,}')
    if totalSubActivity != 0:
        if isinstance(totalSubActivity, float):
            print(f'Total Values: {round(totalSubActivity, 1):,}')
    print()

    # Step 2: Load the ESM model and batch converter
    if paramsESM == '15B Params':
        print(f'Loading Model: esm2_t48_15B_UR50D()')
        model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
        numLayersESM = 48
    else:
        # print(f'Loading Model: esm2_t36_3B_UR50D()')
        # model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        # numLayersESM = 36
        print(f'Loading Model: esm2_t33_650M_UR50D')
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        numLayersESM = 33
    print()
    model = model.to(device)

    # Get: batch tensor
    batchConverter = alphabet.get_batch_converter()

    # Step 3: Convert substrates to ESM model format and generate Embeddings
    try:
        batchLabels, batchSubs, batchTokens = batchConverter(subs)
        batchTokensCPU = batchTokens
        batchTokens = batchTokens.to(device)

    except Exception as exc:
        print(f'ERROR: The ESM has failed to evaluate your substrates\n\n'
              f'Exception:\n{exc}\n\n'
              f'\n\nSuggestion:'
              f'     Try adding more memory:\n'
              f'          #SBATCH --mem=[80G, 100G, 120G, 140G]\n\n')
        sys.exit(1)
    print(f'Batch Tokens: {batchTokens.shape}\n'
          f'{batchTokens}\n')

    # Record tokens
    slicedTokens = pd.DataFrame(batchTokensCPU[:, 1:-1],
                                index=batchSubs,
                                columns=labelsXAxis)
    if totalSubActivity != 0:
        slicedTokens['Values'] = values
    print(f'\nSliced Tokens:\n'
          f'{slicedTokens}\n\n')

    # Generate embeddings
    batchTotal = len(batchTokens)
    allEmbeddings = []
    allValues = []
    startInit = time.time()
    print('Generating ESM Embeddings:')
    with torch.no_grad():
        for i in range(0, len(batchTokens), batchSize):
            start = time.time()
            batch = batchTokens[i:i + batchSize].to(device)
            result = model(batch, repr_layers=[numLayersESM], return_contacts=False)
            tokenReps = result["representations"][numLayersESM]
            seqEmbed = tokenReps[:, 0, :].cpu().numpy()
            allEmbeddings.append(seqEmbed)
            end = time.time()
            runtime = end - start
            runtimeTotal = (end - startInit) / 60
            percentCompletion = round((i / batchTotal) * 100, 1)
            print(f'ESM Progress: {i:,} / {batchTotal:,}'
                  f' ({percentCompletion} %)\n'
                  f'     Batch Shape: {batch.shape}\n'
                  f'     Runtime: {round(runtime, 3):,} s'
                  f'\n'
                  f'     Total Time: {round(runtimeTotal, 3):,} min'
                  f'\n')
            if trainingSet:
                allValues.extend(values[i:i + batchSize])

            # Clear data to help free memory
            del tokenReps, batch
            torch.cuda.empty_cache()
    end = time.time()
    runtime = end - start
    runtimeTotal = (end - startInit) / 60
    percentCompletion = round((batchTotal / batchTotal) * 100, 1)
    print(f'ESM Progress: {batchTotal:,} / {batchTotal:,}'
          f' ({percentCompletion} %)\n'
          f'     Runtime: {round(runtime, 3):,} s'
          f'\n'
          f'     Total Time: {round(runtimeTotal, 3):,} min'
          f'\n')

    # Step 4: Extract per-sequence Embeddings
    tokenReps = result["representations"][numLayersESM]  # (N, seq_len, hidden_dim)
    sequenceEmbeddings = tokenReps[:, 0, :]  # [CLS] token embedding: (N, hidden_dim)

    # Convert to numpy and store substrate activity proxy
    embeddings = np.vstack(allEmbeddings)
    if predictions:
        data = np.hstack([embeddings])
        columns = [f'feat_{i}' for i in range(embeddings.shape[1])]
    else:
        values = np.array(allValues).reshape(-1, 1)
        data = np.hstack([embeddings, values])
        columns = [f'feat_{i}' for i in range(embeddings.shape[1])] + ['activity']

    # Process Embeddings
    subEmbeddings = pd.DataFrame(data, index=batchSubs, columns=columns)
    print(f'\nSubstrate Embeddings:\n{subEmbeddings}\n\n')
    print(f'Saving Embeddings At:\n'
          f'     {pathEmbeddings}\n\n')
    subEmbeddings.to_csv(pathEmbeddings)



# Generate embeddings
ESM(substrates=substrates, paramsESM=modelParams, pathSave=pathEmbeddings,
    trainingSet=True)
# ESM(substrates=substrates, paramsESM=modelParams, pathSave=pathEmbeddings)
