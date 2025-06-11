import esm
import numpy as np
import os
import pandas as pd
import random
import sys
import time
import torch


# Check for input
if len(sys.argv) < 7:
    print("ERROR: Incorrect number of inputs\n"
          "Usage: python convertToJPG.py <input file>")
    sys.exit(1)

# Get params
modelParams = sys.argv[1]
enzymeName = sys.argv[2]
fixAA = sys.argv[3]
fixPos = sys.argv[4]
useReadingFrame = sys.argv[5]
minSubs = sys.argv[6]
batchSize = sys.argv[7]

enzyme = None
if enzymeName.lower() == 'mpro2':
    enzyme = f'SARS-CoV-2 M{'ᵖʳᵒ'}'

# Define file name
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


pathSubs = os.path.join(pathData, fileName)
print(f'Loading Substrates:\n'
      f'     {pathSubs}\n\n')




# sys.exit()


substrates = {
    'AVLQSASA': 125544,
    'TSLQGVFA': 119235,
    'VILQGGTA': 110987,
}



def ESM(self, substrates, paramsESM, trainingSet=False):
    print('=========================== Generate Embeddings: ESM '
          '============================')
    # Choose: ESM PLM model
    modelPrams = 2
    if modelPrams == 0:
        sizeESM = '15B Params'
    elif modelPrams == 1:
        sizeESM = '3B Params'
    else:
        sizeESM = '650M Params'

    # Inspect: Data type
    predictions = True
    if trainingSet:
        predictions = False
    print(f'Dataset: {paramsESM}\n'
          f'Total unique substrates: {len(substrates):,}')

    # Load: ESM Embeddings
    pathEmbeddings = os.path.join(pathEmbeddings, f'{paramsESM}.csv')
    if os.path.exists(pathEmbeddings):
        print(f'\nLoading: ESM Embeddings\n'
              f'     {pathEmbeddings}\n')
        subEmbeddings = pd.read_csv(pathEmbeddings, index_col=0)
        print(f'Substrate Embeddings shape: '
              f'{subEmbeddings.shape}\n\n')

        return subEmbeddings

    # # Generate Embeddings
    # Step 1: Convert substrates to ESM model format and generate Embeddings
    totalSubActivity = 0
    subs = []
    values = []
    if trainingSet:
        # Randomize substrates
        items = list(substrates.items())
        random.shuffle(items)
        substrates = dict(items)

        for index, (substrate, value) in enumerate(substrates.items()):
            totalSubActivity += value
            subs.append((f'Sub{index}', substrate))
            values.append(value)
    else:
        for index, substrate in enumerate(substrates):
            subs.append((f'Sub{index}', substrate))
    sampleSize = len(substrates)
    print(f'Collected substrates: {sampleSize:,}')
    if totalSubActivity != 0:
        if isinstance(totalSubActivity, float):
            print(f'Total Values: {round(totalSubActivity, 1):,}'
                  f'')
        else:
            print(f'Total Values: {totalSubActivity:,}')
    print()

    # Step 2: Load the ESM model and batch converter
    if paramsESM == '15B Params':
        model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
        numLayersESM = 48
    else:
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        numLayersESM = 36
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
              f'Suggestion:'
              f'     Try replacing: esm.pretrained.esm2_t36_3B_UR50D()'
              f'\n'
              f'     With: esm.pretrained.esm2_t33_650M_UR50D()'
              f'\n')
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
          f'{slicedTokens}\n')

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
    print(f'Substrate Embeddings shape: '
          f'{sequenceEmbeddings.shape}\n\n')
    print(f'Embeddings saved at:\n'
          f'     {pathEmbeddings}\n\n')
    subEmbeddings.to_csv(pathEmbeddings)



# Generate embeddings
ESM(substrates=substrates, paramsESM=modelParams, trainingSet=True)
ESM(substrates=substrates, paramsESM=modelParams, trainingSet=True)
