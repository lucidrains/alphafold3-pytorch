import click
from pathlib import Path

import secrets
import shutil
from Bio.PDB import PDBIO

from alphafold3_pytorch import Alphafold3, Alphafold3Input

# constants

model = None
cache_path = None
pdb_writer = PDBIO()

# main fold function

def fold(entities, request):
    proteins = []
    rnas = []
    dnas = []
    ligands = []
    ions = []
    for entity in entities:
        if entity["mol_type"] == "Protein":
            proteins.extend([entity["sequence"]] * entity["num_copies"])
        elif entity["mol_type"] == "RNA":
            rnas.extend([entity["sequence"]] * entity["num_copies"])
        elif entity["mol_type"] == "DNA":
            dnas.extend([entity["sequence"]] * entity["num_copies"])
        elif entity["mol_type"] == "Ligand":
            ligands.extend([entity["sequence"]] * entity["num_copies"])
        elif entity["mol_type"] == "Ion":
            ions.extend([entity["sequence"]] * entity["num_copies"])

    # Prepare the input for the model
    alphafold3_input = Alphafold3Input(
        proteins=proteins,
        ss_dna=dnas,
        ss_rna=rnas,
        ligands=ligands,
        metal_ions=ions,
    )

    # Run the model inference in a separate thread
    model.eval()
    (structure,) = model.forward_with_alphafold3_inputs(
        alphafold3_inputs=alphafold3_input,
        return_bio_pdb_structures=True,
    )

    global cache_path, pdb_writer
    output_path = cache_path / str(request.session_hash) / f"{secrets.token_urlsafe(8)}.pdb"
    output_path.parent.mkdir(exist_ok=True)

    pdb_writer.set_structure(structure)
    pdb_writer.save(str(output_path))

    return str(output_path)

# gradio

def delete_cache(request):
    if not request.session_hash:
        return

    user_dir: Path = cache_path / request.session_hash
    if user_dir.exists():
        shutil.rmtree(str(user_dir))


def start_gradio_app():
    import gradio as gr
    from gradio_molecule3d import Molecule3D

    with gr.Blocks(delete_cache=(600, 3600)) as gradio_app:
        entities = gr.State([])

        with gr.Row():
            gr.Markdown("### AlphaFold3 PyTorch Web UI")

        with gr.Row():
            gr.Column(scale=8)
            # upload_json_button = gr.Button("Upload JSON", scale=1, min_width=100)
            clear_button = gr.Button("Clear", scale=1, min_width=100)

        with gr.Row():
            with gr.Column(scale=1, min_width=150):
                mtype = gr.Dropdown(
                    value="Protein",
                    label="Molecule type",
                    choices=["Protein", "DNA", "RNA", "Ligand", "Ion"],
                    interactive=True,
                )
            with gr.Column(scale=1, min_width=80):
                c = gr.Number(
                    value=1,
                    label="Copies",
                    interactive=True,
                )

            with gr.Column(scale=8, min_width=200):

                @gr.render(inputs=mtype)
                def render_sequence(mol_type):
                    if mol_type in ["Protein", "DNA", "RNA"]:
                        seq = gr.Textbox(
                            label="Paste sequence or fasta",
                            placeholder="Input",
                            interactive=True,
                        )
                    elif mol_type == "Ligand":
                        seq = gr.Dropdown(
                            label="Select ligand",
                            choices=[
                                "ADP - Adenosine disphosphate",
                                "ATP - Adenosine triphosphate",
                                "AMP - Adenosine monophosphate",
                                "GTP - Guanosine-5'-triphosphate",
                                "GDP - Guanosine-5'-diphosphate",
                                "FAD - Flavin adenine dinucleotide",
                                "NAD - Nicotinamide-adenine-dinucleotide",
                                "NAP - Nicotinamide-adenine-dinucleotide phosphate (NADP)",
                                "NDP - Dihydro-nicotinamide-adenine-dinucleotide-phosphate (NADPH)",
                                "HEM - Heme",
                                "HEC - Heme C",
                                "OLA - Oleic acid",
                                "MYR - Myristic acid",
                                "CIT - Citric acid",
                                "CLA - Chlorophyll A",
                                "CHL - Chlorophyll B",
                                "BCL - Bacteriochlorophyll A",
                                "BCB - Bacteriochlorophyll B",
                            ],
                            interactive=True,
                        )
                    elif mol_type == "Ion":
                        seq = gr.Dropdown(
                            label="Select ion",
                            choices=[
                                "Mg²⁺",
                                "Zn²⁺",
                                "Cl⁻",
                                "Ca²⁺",
                                "Na⁺",
                                "Mn²⁺",
                                "K⁺",
                                "Fe³⁺",
                                "Cu²⁺",
                                "Co²⁺",
                            ],
                            interactive=True,
                        )

                    add_button.click(add_entity, inputs=[entities, mtype, c, seq], outputs=[entities])
                    clear_button.click(lambda: ("Protein", 1, None), None, outputs=[mtype, c, seq])

        add_button = gr.Button("Add entity", scale=1, min_width=100)

        def add_entity(entities, mtype="Protein", c=1, seq=""):
            if seq is None or len(seq) == 0:
                gr.Info("Input required")
                return entities

            seq_norm = seq.strip(" \t\n\r").upper()

            if mtype in ["Protein", "DNA", "RNA"]:
                if mtype == "Protein" and any([x not in "ARDCQEGHILKMNFPSTWYV" for x in seq_norm]):
                    gr.Info("Invalid protein sequence. Allowed characters: A, R, D, C, Q, E, G, H, I, L, K, M, N, F, P, S, T, W, Y, V")
                    return entities

                if mtype == "DNA" and any([x not in "ACGT" for x in seq_norm]):
                    gr.Info("Invalid DNA sequence. Allowed characters: A, C, G, T")
                    return entities

                if mtype == "RNA" and any([x not in "ACGU" for x in seq_norm]):
                    gr.Info("Invalid RNA sequence. Allowed characters: A, C, G, U")
                    return entities

                if len(seq) < 4:
                    gr.Info("Minimum 4 characters required")
                    return entities

            elif mtype == "Ligand":
                if seq is None or len(seq) == 0:
                    gr.Info("Select a ligand")
                    return entities
                seq_norm = seq.split(" - ")[0]
            elif mtype == "Ion":
                if seq is None or len(seq) == 0:
                    gr.Info("Select an ion")
                    return entities
                seq_norm = "".join([x for x in seq if x.isalpha()])

            new_entity = {"mol_type": mtype, "num_copies": c, "sequence": seq_norm}

            return entities + [new_entity]

        @gr.render(inputs=entities)
        def render_entities(entity_list):
            for idx, entity in enumerate(entity_list):
                with gr.Row():
                    gr.Text(
                        value=entity["mol_type"],
                        label="Type",
                        scale=1,
                        min_width=90,
                        interactive=False,
                    )
                    gr.Text(
                        value=entity["num_copies"],
                        label="Copies",
                        scale=1,
                        min_width=80,
                        interactive=False,
                    )

                    sequence = entity["sequence"]
                    if entity["mol_type"] not in ["Ligand", "Ion"]:
                        # Split every 10 characters, and add a \t after each split
                        sequence = "\t".join([sequence[i : i + 10] for i in range(0, len(sequence), 10)])

                    gr.Text(
                        value=sequence,
                        label="Sequence",
                        placeholder="Input",
                        scale=7,
                        min_width=200,
                        interactive=False,
                    )

                    del_button = gr.Button("🗑️", scale=0, min_width=50)

                    def delete(entity_id=idx):
                        entity_list.pop(entity_id)
                        return entity_list

                    del_button.click(delete, None, outputs=[entities])

        pred_button = gr.Button("Predict", scale=1, min_width=100)
        output_mol = Molecule3D(label="Output structure", config={"backgroundColor": "black"})

        pred_button.click(fold, inputs=entities, outputs=output_mol)
        clear_button.click(lambda: ([], None), None, outputs=[entities, output_mol])

        gradio_app.unload(delete_cache)
        gradio_app.launch()

# cli
@click.command()
@click.option("-ckpt", "--checkpoint", type=str, help="path to alphafold3 checkpoint", required=True)
@click.option("-cache", "--cache-dir", type=str, help="path to output cache", required=False, default="cache")
@click.option("-prec", "--precision", type=str, help="precision to use", required=False, default="float32")
def app(checkpoint: str, cache_dir: str, precision: str):
    path = Path(checkpoint)
    assert path.exists(), "checkpoint does not exist at path"

    global cache_path
    cache_path = Path(cache_dir)

    if cache_path.exists():
        shutil.rmtree(str(cache_path))

    cache_path.mkdir(exist_ok=True)

    global model
    model = Alphafold3.init_and_load(str(path))
    # To device and quantize?
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # try:
    #     dtype = getattr(torch, precision)
    # except AttributeError:
    #     print(f"Invalid precision: {precision}. Using float32")
    #     dtype = torch.float32
    # model.to(device, dtype=dtype)

    start_gradio_app()
