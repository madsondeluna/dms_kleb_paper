#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 14:56:51 2023

@author: joao
"""
"""
rosetta_utils.py
================
Utility functions for protein structure modeling with PyRosetta, covering:

  - Pose utilities and PDB/Pose residue index mapping
  - Relaxation and side-chain packing protocols
  - Single-point mutation with neighbor repacking
  - Per-residue and per-term energy decomposition
  - Binding free energy (dG) calculation
  - Interface descriptors (interaction energy, CMS, InterfaceAnalyzerMover)
  - In silico Deep Mutational Scanning (DMS): ΔΔG per energy term, parallel via multiprocessing.Pool
  - Sequence modeling: load, compare, mutate, and export structures
  - I/O helpers for JD2-compatible PDB files

Authors: joao
Last revised: 2026
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import os
import sys
import pandas as pd

# ---------------------------------------------------------------------------
# PyRosetta
# ---------------------------------------------------------------------------
import pyrosetta
from pyrosetta import rosetta, pose_from_pdb

from rosetta.core.kinematics import MoveMap, FoldTree
from rosetta.core.pack.task import TaskFactory, operation
from rosetta.core.simple_metrics import metrics
from rosetta.core.select import residue_selector as selections
from rosetta.core import select
from rosetta.core.select.movemap import *
from rosetta.protocols import minimization_packing as pack_min
from rosetta.protocols import relax as rel
from rosetta.protocols.antibody.residue_selector import CDRResidueSelector
from rosetta.protocols.antibody import *
from rosetta.protocols.loops import *
from rosetta.protocols.relax import FastRelax
from rosetta.protocols.docking import setup_foldtree
from rosetta.protocols import *
from rosetta.core.scoring.methods import EnergyMethodOptions
from pyrosetta.rosetta.protocols.docking import setup_foldtree as pyros_setup_foldtree
from pyrosetta.rosetta.protocols import *


# ===========================================================================
# SECTION 1 — Pose Utilities
# ===========================================================================

def PDB_pose_dictionairy(pose):
    """
    Build a mapping between PDB residue numbering and Rosetta Pose numbering.

    Iterates over every residue in the pose and records its chain identifier,
    original PDB sequence number, and the internal Pose index used by Rosetta.
    The resulting DataFrame is used by downstream functions to translate between
    the two numbering systems.

    Parameters
    ----------
    pose : pyrosetta.Pose
        Rosetta Pose object loaded from a PDB file.

    Returns
    -------
    pd.DataFrame
        Three-column DataFrame with the following fields:

        - **Chain** (*str*): single-letter chain identifier (e.g. ``'A'``).
        - **IndexPDB** (*int*): residue number as it appears in the PDB file.
        - **IndexPose** (*int*): 1-based Rosetta internal residue index.
    """
    chains, pdb_ids, pose_ids = [], [], []
    for i in range(pose.total_residue()):
        chain = pose.pdb_info().chain(i + 1)
        pdb_num = pose.pdb_info().number(i + 1)
        pose_num = pose.pdb_info().pdb2pose(chain, pdb_num)
        chains.append(chain)
        pdb_ids.append(pdb_num)
        pose_ids.append(pose_num)

    return pd.DataFrame({
        "Chain": chains,
        "IndexPDB": pdb_ids,
        "IndexPose": pose_ids,
    })


def residues_list(df, chain):
    """
    Return the list of Pose indices that belong to a given chain.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame produced by :func:`PDB_pose_dictionairy`.
    chain : str
        Single-letter chain identifier (e.g. ``'A'``).

    Returns
    -------
    list[int]
        Rosetta Pose indices (1-based) for all residues in *chain*.
    """
    return list(df[df["Chain"] == chain]["IndexPose"])


# ===========================================================================
# SECTION 2 — Relaxation and Packing Protocols
# ===========================================================================

def pack_relax(pose, scorefxn):
    """
    Apply a Cartesian FastRelax protocol with a free backbone to the pose.

    The protocol uses a fully permissive MoveMapFactory (all backbone torsions,
    bond angles, bond lengths, side-chain chi angles, and jumps are moveable)
    combined with a repacking-only TaskFactory. Minimisation is performed with
    the L-BFGS Armijo non-monotone line search.

    Parameters
    ----------
    pose : pyrosetta.Pose
        Pose to be relaxed. Modified **in place**.
    scorefxn : pyrosetta.ScoreFunction
        Score function used during the FastRelax run (should be a Cartesian-
        compatible score function such as ``ref2015_cart``).
    """
    tf = TaskFactory()
    tf.push_back(operation.InitializeFromCommandline())
    tf.push_back(operation.RestrictToRepacking())

    mmf = pyrosetta.rosetta.core.select.movemap.MoveMapFactory()
    mmf.all_bb(setting=True)
    mmf.all_bondangles(setting=True)
    mmf.all_bondlengths(setting=True)
    mmf.all_chi(setting=True)
    mmf.all_jumps(setting=True)
    mmf.set_cartesian(setting=True)

    fr = pyrosetta.rosetta.protocols.relax.FastRelax(scorefxn_in=scorefxn, standard_repeats=1)
    fr.cartesian(True)
    fr.set_task_factory(tf)
    fr.set_movemap_factory(mmf)
    fr.min_type("lbfgs_armijo_nonmonotone")
    fr.apply(pose)


def minimize(pose, scorefxn, minimizer_type):
    """
    Perform energy minimisation on the pose using Rosetta's MinMover.

    Two minimisation modes are available, differing in which degrees of freedom
    are optimised:

    - ``'minmover1'``: side-chain chi angles only (backbone fixed). Useful for
      fast rotamer optimisation without disturbing the backbone.
    - ``'minmover2'``: backbone phi/psi torsions **and** side-chain chi angles.
      Produces a more thorough optimisation but is computationally heavier.

    Both modes use a tolerance of ``1e-4`` and allow up to 50 000 iterations.
    Internal coordinates (torsional) minimisation is used (Cartesian off).

    Parameters
    ----------
    pose : pyrosetta.Pose
        Pose to minimise. Modified **in place**.
    scorefxn : pyrosetta.ScoreFunction
        Score function used during minimisation.
    minimizer_type : {'minmover1', 'minmover2'}
        Selects the degree-of-freedom set as described above.

    Raises
    ------
    ValueError
        If *minimizer_type* is not one of the accepted values.
    """
    movemap = pyrosetta.rosetta.core.kinematics.MoveMap()
    if minimizer_type == "minmover1":
        movemap.set_bb(False)
        movemap.set_chi(True)
    elif minimizer_type == "minmover2":
        movemap.set_bb(True)
        movemap.set_chi(True)
    else:
        raise ValueError(
            f"Unknown minimizer_type '{minimizer_type}'. "
            "Expected 'minmover1' or 'minmover2'."
        )

    min_mover = rosetta.protocols.minimization_packing.MinMover()
    min_mover.score_function(scorefxn)
    min_mover.max_iter(50000)
    min_mover.tolerance(0.0001)
    min_mover.cartesian(False)
    min_mover.movemap(movemap)
    min_mover.apply(pose)


# ===========================================================================
# SECTION 3 — Mutation and Repacking
# ===========================================================================

def mutate_repack(starting_pose, posi, amino, scorefxn):
    """
    Introduce a point mutation at a given Pose position and repack neighbours.

    The function clones the input pose so that the original is never modified.
    The mutation is applied via a design-restricted TaskFactory that:

    1. Allows rotamer sampling only within the neighbourhood of the target
       residue (atoms within the shell defined by
       ``NeighborhoodResidueSelector``).
    2. Restricts the target residue to the single canonical amino acid
       specified by *amino*.
    3. Restricts all other neighbourhood residues to repacking only
       (no sequence change).
    4. Freezes residues outside the neighbourhood entirely.

    Disulfide bonds are preserved (``NoRepackDisulfides``).

    Parameters
    ----------
    starting_pose : pyrosetta.Pose
        Template pose. Not modified.
    posi : int
        Rosetta Pose index (1-based) of the residue to mutate.
    amino : str
        One-letter code of the target amino acid (e.g. ``'A'`` for alanine).
    scorefxn : pyrosetta.ScoreFunction
        Score function used for rotamer packing.

    Returns
    -------
    pyrosetta.Pose
        A new pose carrying the requested mutation with repacked neighbours.
    """
    pose = starting_pose.clone()

    # --- Residue selectors ---------------------------------------------------
    mut_posi = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
    mut_posi.set_index(posi)

    nbr_selector = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector()
    nbr_selector.set_focus_selector(mut_posi)
    nbr_selector.set_include_focus_in_subset(True)

    not_design = pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector(mut_posi)

    # --- Task factory --------------------------------------------------------
    tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.NoRepackDisulfides())

    # Freeze residues outside the neighbourhood
    prevent_rlt = pyrosetta.rosetta.core.pack.task.operation.PreventRepackingRLT()
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
        prevent_rlt, nbr_selector, True))

    # Repack-only for non-target residues inside the neighbourhood
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
        pyrosetta.rosetta.core.pack.task.operation.RestrictToRepackingRLT(),
        not_design))

    # Restrict the target position to the requested amino acid
    aa_to_design = pyrosetta.rosetta.core.pack.task.operation.RestrictAbsentCanonicalAASRLT()
    aa_to_design.aas_to_keep(amino)
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
        aa_to_design, mut_posi))

    # --- Packing -------------------------------------------------------------
    packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(scorefxn)
    packer.task_factory(tf)
    packer.apply(pose)

    return pose


# ===========================================================================
# SECTION 4 — Energy Calculation
# ===========================================================================

def _build_scorefxn_with_hbond():
    """
    Create a ``ref2015_cart`` score function with hydrogen-bond decomposition.

    Backbone hydrogen-bond energies are split into pairwise contributions
    (``decompose_bb_hb_into_pair_energies = True``), which is required for
    correct per-residue energy reporting.

    Returns
    -------
    pyrosetta.ScoreFunction
        Configured ``ref2015_cart`` score function.
    """
    scorefxn = pyrosetta.create_score_function("ref2015_cart.wts")
    emopts = EnergyMethodOptions(scorefxn.energy_method_options())
    emopts.hbond_options().decompose_bb_hb_into_pair_energies(True)
    scorefxn.set_energy_method_options(emopts)
    return scorefxn


def Energy_contribution(pdb, by_term=True):
    """
    Compute per-residue energy contributions for a given pose.

    A fresh ``ref2015_cart`` score function with hydrogen-bond decomposition
    is created internally and applied to the pose before extracting energies,
    ensuring that the energy graph is up to date.

    Parameters
    ----------
    pose : pdb_path
        PDB to analyse.
    by_term : bool, optional
        - ``True`` *(default)*: returns a wide DataFrame where each column
          corresponds to a weighted energy term (only terms with non-zero
          weights are included) and each row corresponds to one residue.
          Additional metadata columns are prepended (see *Returns*).
        - ``False``: returns a single-row DataFrame (transposed) containing
          only the total energy per residue, indexed by Pose position.

    Returns
    -------
    pd.DataFrame
        **When** ``by_term=True``:

        Columns (in order):

        - ``Residue_Index_Pose`` (*int*): 1-based Rosetta internal index.
        - ``Residue_Index_PDB`` (*int*): original PDB sequence number.
        - ``Residue_Name`` (*str*): full Rosetta residue name
          (e.g. ``'ARG:NtermProteinFull'``).
        - ``Residue_Name1`` (*str*): standard one-letter amino acid code.
        - ``Chain`` (*str*): chain identifier.
        - *<energy_term>* (*float*): weighted energy contribution for every
          active score term (e.g. ``fa_atr``, ``hbond_sc``, ``fa_elec``).

        **When** ``by_term=False``:

        A transposed single-row DataFrame where column indices are
        0-based integer positions (0, 1, ..., N-1) corresponding to
        residues in Pose order; the single row contains total weighted
        energies. Note: columns are NOT 1-based Pose indices.

    Notes
    -----
    Energy values are **weighted** (unweighted raw value x score-function
    weight), matching the contributions that appear in Rosetta score files.
    """

    pose = read_pose(pdb)[0]
    
    AA_3TO1 = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
        "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
        "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
        "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
        "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }

    scorefxn = _build_scorefxn_with_hbond()
    scorefxn.score(pose)
    weights = scorefxn.weights()
    index_df = PDB_pose_dictionairy(pose)

    # Collect only score types that carry a non-zero weight
    active_score_types = [
        st for st in rosetta.core.scoring.ScoreType.__members__.values()
        if weights[st] != 0
    ]

    if not by_term:
        residues = [res.seqpos() for res in pose]
        total_energies = [pose.energies().residue_total_energy(r) for r in residues]
        return pd.DataFrame(total_energies).T

    data = []
    for i in range(1, pose.total_residue() + 1):
        res_energies = pose.energies().residue_total_energies(i)
        row = {
            "Residue_Index_Pose": i,
            "Residue_Name": pose.residue(i).name(),
        }
        for st in active_score_types:
            row[st.name] = res_energies[st] * weights[st]
        data.append(row)

    df = pd.DataFrame(data)
    df["Residue_Name1"] = df["Residue_Name"].str[:3].map(AA_3TO1)
    df["Residue_Index_PDB"] = index_df["IndexPDB"].values
    df["Chain"] = index_df["Chain"].values

    priority_cols = [
        "Residue_Index_Pose", "Residue_Index_PDB",
        "Residue_Name", "Residue_Name1", "Chain",
    ]
    other_cols = [c for c in df.columns if c not in priority_cols]
    return df[priority_cols + other_cols]

def Energy_contribution_DMS(pose, by_term=True):
    """
    Compute per-residue energy contributions for a given pose.

    A fresh ``ref2015_cart`` score function with hydrogen-bond decomposition
    is created internally and applied to the pose before extracting energies,
    ensuring that the energy graph is up to date.

    Parameters
    ----------
    pose : pose.object
        Pose to analyse.
    by_term : bool, optional
        - ``True`` *(default)*: returns a wide DataFrame where each column
          corresponds to a weighted energy term (only terms with non-zero
          weights are included) and each row corresponds to one residue.
          Additional metadata columns are prepended (see *Returns*).
        - ``False``: returns a single-row DataFrame (transposed) containing
          only the total energy per residue, indexed by Pose position.

    Returns
    -------
    pd.DataFrame
        **When** ``by_term=True``:

        Columns (in order):

        - ``Residue_Index_Pose`` (*int*): 1-based Rosetta internal index.
        - ``Residue_Index_PDB`` (*int*): original PDB sequence number.
        - ``Residue_Name`` (*str*): full Rosetta residue name
          (e.g. ``'ARG:NtermProteinFull'``).
        - ``Residue_Name1`` (*str*): standard one-letter amino acid code.
        - ``Chain`` (*str*): chain identifier.
        - *<energy_term>* (*float*): weighted energy contribution for every
          active score term (e.g. ``fa_atr``, ``hbond_sc``, ``fa_elec``).

        **When** ``by_term=False``:

        A transposed single-row DataFrame where column indices are
        0-based integer positions (0, 1, ..., N-1) corresponding to
        residues in Pose order; the single row contains total weighted
        energies. Note: columns are NOT 1-based Pose indices.

    Notes
    -----
    Energy values are **weighted** (unweighted raw value x score-function
    weight), matching the contributions that appear in Rosetta score files.
    """
    
    
    AA_3TO1 = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
        "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
        "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
        "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
        "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }

    scorefxn = _build_scorefxn_with_hbond()
    scorefxn.score(pose)
    weights = scorefxn.weights()
    index_df = PDB_pose_dictionairy(pose)

    # Collect only score types that carry a non-zero weight
    active_score_types = [
        st for st in rosetta.core.scoring.ScoreType.__members__.values()
        if weights[st] != 0
    ]

    if not by_term:
        residues = [res.seqpos() for res in pose]
        total_energies = [pose.energies().residue_total_energy(r) for r in residues]
        return pd.DataFrame(total_energies).T

    data = []
    for i in range(1, pose.total_residue() + 1):
        res_energies = pose.energies().residue_total_energies(i)
        row = {
            "Residue_Index_Pose": i,
            "Residue_Name": pose.residue(i).name(),
        }
        for st in active_score_types:
            row[st.name] = res_energies[st] * weights[st]
        data.append(row)

    df = pd.DataFrame(data)
    df["Residue_Name1"] = df["Residue_Name"].str[:3].map(AA_3TO1)
    df["Residue_Index_PDB"] = index_df["IndexPDB"].values
    df["Chain"] = index_df["Chain"].values

    priority_cols = [
        "Residue_Index_Pose", "Residue_Index_PDB",
        "Residue_Name", "Residue_Name1", "Chain",
    ]
    other_cols = [c for c in df.columns if c not in priority_cols]
    return df[priority_cols + other_cols]


def Get_energy_per_term(pose, scorefxn):
    """
    Return a dictionary of total weighted energy terms for the entire pose.

    Only terms with a non-zero weight in *scorefxn* are included. The score
    function must have been applied to the pose before calling this function
    (i.e. ``scorefxn(pose)`` should be called beforehand) so that the energy
    graph is current.

    Parameters
    ----------
    pose : pyrosetta.Pose
        Pose whose total energies will be extracted.
    scorefxn : pyrosetta.ScoreFunction
        Score function that was used to score the pose.

    Returns
    -------
    dict[str, float]
        Mapping of ``{score_term_name: weighted_energy_value}``.
        Example keys: ``'fa_atr'``, ``'fa_rep'``, ``'hbond_sc'``, etc.
    """
    weights = scorefxn.weights()
    energy_map = pose.energies().total_energies()
    return {
        st.name: energy_map[st] * weights[st]
        for st in rosetta.core.scoring.ScoreType.__members__.values()
        if weights[st] != 0
    }


# ===========================================================================
# SECTION 5 — Binding Free Energy (dG)
# ===========================================================================

def unbind(pose, partners, scorefxn):
    """
    Separate the chains of a complex and relax the unbound state.

    The function clones the input pose, translates one partner 100 A away
    along the interface jump vector (using ``RigidBodyTransMover``), and then
    applies :func:`pack_relax` to the separated complex to relieve steric
    clashes introduced by the translation.

    Parameters
    ----------
    pose : pyrosetta.Pose
        Bound complex. Not modified.
    partners : str
        Partner string in Rosetta docking notation (e.g. ``'A_B'`` to separate
        chain A from chain B, or ``'AB_C'`` for a heterodimer vs. a third
        chain).
    scorefxn : pyrosetta.ScoreFunction
        Score function used during the relaxation of the unbound state.

    Returns
    -------
    tuple[pyrosetta.Pose, pyrosetta.Pose]
        A two-element tuple ``(pose_bound, pose_unbound)`` where:

        - ``pose_bound``: clone of the original bound complex.
        - ``pose_unbound``: relaxed separated complex after translation.
    """
    pose_bound = pose.clone()
    pose_unbound = pose.clone()

    STEP_SIZE = 100
    JUMP = 1
    pyros_setup_foldtree(pose_unbound, partners, Vector1([-1, -1, -1]))
    trans_mover = rigid.RigidBodyTransMover(pose_unbound, JUMP)
    trans_mover.step_size(STEP_SIZE)
    trans_mover.apply(pose_unbound)
    pack_relax(pose_unbound, scorefxn)

    return pose_bound, pose_unbound


def dG_binding(pose, partners, scorefxn):
    """
    Calculate the binding free energy (dG_bind) of a protein complex.

    dG_bind is estimated as the difference between the Rosetta total score of
    the bound state and the relaxed unbound state:

        dG = E(bound) - E(unbound)

    A negative value indicates a favourable (stabilising) interaction.

    Parameters
    ----------
    pose : pyrosetta.Pose
        Protein complex in the bound conformation. Not modified (internally
        cloned before unbinding).
    partners : str
        Partner string in Rosetta docking notation (e.g. ``'A_B'``).
        See :func:`unbind` for details.
    scorefxn : pyrosetta.ScoreFunction
        Score function used for both the bound and unbound energy evaluations.

    Returns
    -------
    float
        dG_bind in Rosetta energy units (REU).
    """
    pose_bound, pose_unbound = unbind(pose.clone(), partners, scorefxn)
    return scorefxn(pose_bound) - scorefxn(pose_unbound)


# ===========================================================================
# SECTION 6 — Interface Descriptors
# ===========================================================================

def Interaction_energy_metric(pose, scorefxn, partner1, partner2):
    """
    Compute the pairwise interaction energy between two groups of chains.

    Uses Rosetta's ``InteractionEnergyMetric`` simple metric, which evaluates
    only the cross-partner energy terms (i.e. residue pairs where one residue
    belongs to *partner1* and the other to *partner2*).

    Parameters
    ----------
    pose : pyrosetta.Pose
        Scored protein complex.
    scorefxn : pyrosetta.ScoreFunction
        Score function to use for the interaction energy calculation.
    partner1 : str
        Chain letters for the first partner (e.g. ``'A'`` or ``'AB'``).
    partner2 : str
        Chain letters for the second partner (e.g. ``'B'`` or ``'C'``).

    Returns
    -------
    float
        Interaction energy in Rosetta energy units (REU).
    """
    def _chain_selector(chains):
        sel = rosetta.core.select.residue_selector.OrResidueSelector()
        for c in chains:
            sel.add_residue_selector(
                rosetta.core.select.residue_selector.ChainSelector(c))
        return sel

    metric = rosetta.core.simple_metrics.metrics.InteractionEnergyMetric()
    metric.set_scorefunction(scorefxn)
    metric.set_residue_selectors(_chain_selector(partner1), _chain_selector(partner2))
    return metric.calculate(pose)


def Contact_molecular_surface(pose, partner1, partner2):
    """
    Calculate the Contact Molecular Surface (CMS) between two binding partners.

    The CMS is a geometric descriptor that estimates the buried surface area
    at the interface, weighted by inter-residue proximity (distance weight
    = 0.5). It is computed using Rosetta's ``ContactMolecularSurfaceFilter``.

    Parameters
    ----------
    pose : pyrosetta.Pose
        Protein complex.
    partner1 : str
        Chain letters for the first partner (e.g. ``'A'``).
    partner2 : str
        Chain letters for the second partner (e.g. ``'B'``).

    Returns
    -------
    float
        Contact molecular surface value (arbitrary Rosetta units).
    """
    def _chain_selector(chains):
        sel = rosetta.core.select.residue_selector.OrResidueSelector()
        for c in chains:
            sel.add_residue_selector(
                rosetta.core.select.residue_selector.ChainSelector(c))
        return sel

    cms = rosetta.protocols.simple_filters.ContactMolecularSurfaceFilter()
    cms.selector1(_chain_selector(partner1))
    cms.selector2(_chain_selector(partner2))
    cms.distance_weight(0.5)
    cms.set_user_defined_name("cms")
    cms.apply(pose)
    return cms.score(pose)


def Interface_analyzer_mover(pose, partner1, partner2):
    """
    Apply Rosetta's InterfaceAnalyzerMover and return all interface metrics.

    The mover computes a comprehensive set of interface properties including
    dG_separated, dG_cross, buried SASA, packing statistics, number of
    interface residues, and more. All metrics are stored on the pose score
    map with the prefix ``'ifa_'`` and returned as a flat dictionary.

    Parameters
    ----------
    pose : pyrosetta.Pose
        Protein complex. Modified **in place** (score map entries are added).
    partner1 : str
        Chain letters for the first partner (e.g. ``'A'``).
    partner2 : str
        Chain letters for the second partner (e.g. ``'B'``).

    Returns
    -------
    dict[str, float]
        Dictionary of interface metrics keyed by their ``'ifa_'``-prefixed
        names (e.g. ``'ifa_dG_separated'``, ``'ifa_packstat'``).
    """
    partners = f"{partner1}_{partner2}"
    ifa = rosetta.protocols.analysis.InterfaceAnalyzerMover(partners)
    ifa.set_use_tracer(True)
    ifa.set_compute_packstat(True)
    ifa.set_scorefile_reporting_prefix("ifa")
    ifa.apply(pose)
    ifa.add_score_info_to_pose(pose)
    return {k: v for k, v in pose.scores.items() if k.startswith("ifa")}


def Get_interface_selector(pose, partner1, partner2):
    """
    Build a ResidueIndexSelector containing all interface residues.

    Uses ``InterfaceAnalyzerMover`` to identify residues that form contacts
    across the partner interface, then packages them into a
    ``ResidueIndexSelector`` that can be used in subsequent TaskFactory or
    MoveMap operations.

    Parameters
    ----------
    pose : pyrosetta.Pose
        Protein complex.
    partner1 : str
        Chain letters for the first partner (e.g. ``'A'``).
    partner2 : str
        Chain letters for the second partner (e.g. ``'B'``).

    Returns
    -------
    pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector
        Selector pre-loaded with the Pose indices of all interface residues.
    """
    partners = f"{partner1}_{partner2}"
    ifa = rosetta.protocols.analysis.InterfaceAnalyzerMover(partners)
    ifa.set_use_tracer(True)
    ifa.set_compute_packstat(True)
    ifa.set_scorefile_reporting_prefix("ifa")
    ifa.apply(pose)
    interface_data = ifa.get_all_data()

    residues_in_interface = [
        i for i in range(1, len(interface_data.interface_residues[1]) + 1)
        if interface_data.interface_residues[1][i]
    ]
    selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
    selector.set_index(",".join(map(str, residues_in_interface)))
    return selector


# ===========================================================================
# SECTION 7 — Full Descriptor Pipeline
# ===========================================================================

def Get_Interface_descriptors(pdb, partner1, partner2, minimize = False):
    """
    Load a structure, minimise it, and compute a full set of interface descriptors.

    This is the main end-to-end pipeline function. It performs the following
    steps in order:

    1. Initialise PyRosetta with ``beta_nov16`` corrections and optional metal
       ion support.
    2. Load the PDB file and create a ``beta_nov16`` score function.
    3. Remap chain identifiers from the user-supplied partner strings to the
       actual chain labels found in the pose (accounts for JD2 renumbering).
    4. Run two rounds of energy minimisation: side-chain only (``minmover1``)
       followed by backbone + side-chain (``minmover2``).
    5. Compute all descriptors: per-term energies, interaction energy (IE),
       contact molecular surface (CMS), and InterfaceAnalyzerMover metrics.
    6. Consolidate results into a single-row DataFrame.

    Rosetta output (tracer, core logs) is silenced during execution and
    restored afterwards, even if an exception is raised.

    Parameters
    ----------
    pdb : str
        Path to the input PDB file.
    partner1 : str
        Chain letters defining the first binding partner (e.g. ``'A'``).
    partner2 : str
        Chain letters defining the second binding partner (e.g. ``'B'``).
    minimize : bool
        Runs minmover before IFA calculation (False by default, assuming the protocol is being used in a previously relaxed structure)

    Returns
    -------
    tuple[pyrosetta.Pose, pd.DataFrame]
        - **pose**: the minimised Pose after both minimisation rounds.
        - **df**: single-row DataFrame containing all computed descriptors.
          Columns prefixed with ``'ifa_'`` originate from
          InterfaceAnalyzerMover; ``'cms'`` and ``'interaction_energy'``
          are the CMS and IE values respectively; the remaining columns are
          weighted per-term energies from the ``beta_nov16`` score function.

    Notes
    -----
    Columns ``'ifa_dG_separated/dSASAx100'`` and ``'ifa_dG_cross/dSASAx100'``
    are automatically renamed to ``'ifa_dG_separated_dSASAx100'`` and
    ``'ifa_dG_cross_dSASAx100'`` to avoid issues with special characters in
    downstream tooling.
    """
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

    try:
        extra = (
            "-mute core -mute basic "
            "-ex1 -ex2 -ex1aro -ex2aro "
            "-use_input_sc -flip_HNQ -no_optH false "
            "-corrections::beta_nov16 true "
            "-output_pose_energies_table false"
        )

        pyrosetta.init(extra_options=extra)
        pose = rosetta.core.import_pose.pose_from_file(pdb)
        scorefxn = rosetta.core.scoring.ScoreFunctionFactory.create_score_function("beta_nov16")
        scorefxn(pose)

        # Map user-supplied chain letters to the actual chains in the pose
        # (necessary when JD2 renumbers chains sequentially)
        all_old_chains = list(partner1) + list(partner2)
        chains_in_pose = list(dict.fromkeys(
            pose.pdb_info().chain(i) for i in range(1, pose.size() + 1)
        ))
        chain_map = dict(zip(all_old_chains, chains_in_pose))
        partner1 = "".join(chain_map[c] for c in partner1)
        partner2 = "".join(chain_map[c] for c in partner2)
        
        if minimize:
            minimize(pose, scorefxn, "minmover1")
            minimize(pose, scorefxn, "minmover2")
            scorefxn(pose)

        per_term = Get_energy_per_term(pose, scorefxn)
        ie       = Interaction_energy_metric(pose, scorefxn, partner1, partner2)
        cms      = Contact_molecular_surface(pose, partner1, partner2)
        ifa      = Interface_analyzer_mover(pose, partner1, partner2)

        all_terms = {**per_term, **ifa}
        all_terms["cms"] = cms
        all_terms["interaction_energy"] = ie
        all_terms["total_score"] = scorefxn(pose)

        df = pd.DataFrame([all_terms])
        df.rename(columns={
            "ifa_dG_separated/dSASAx100": "ifa_dG_separated_dSASAx100",
            "ifa_dG_cross/dSASAx100":     "ifa_dG_cross_dSASAx100",
        }, inplace=True)

    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    return df


# ===========================================================================
# SECTION 8 — In Silico Deep Mutational Scanning (DMS)
# ===========================================================================

import multiprocessing
import logging

log = logging.getLogger(__name__)

# All 20 canonical amino acids (one-letter codes)
_AA_20 = ['G', 'A', 'L', 'M', 'F', 'W', 'K', 'Q', 'E', 'S',
          'P', 'V', 'I', 'C', 'Y', 'H', 'R', 'N', 'D', 'T']

_AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def fast_relax(pose, scorefxn, repeats: int = 1) -> None:
    """
    Apply a Cartesian FastRelax protocol to the pose in place.

    Wraps the same MoveMapFactory / TaskFactory setup used by
    :func:`pack_relax`, but exposes a *repeats* parameter so callers can
    trade runtime for geometric quality.

    Parameters
    ----------
    pose : pyrosetta.Pose
        Pose to relax. Modified **in place**.
    scorefxn : pyrosetta.ScoreFunction
        ``ref2015_cart``-compatible score function.
    repeats : int, optional
        Number of FastRelax rounds (``standard_repeats``). Default: ``1``.
        Higher values improve backbone geometry but increase runtime linearly.
    """
    tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.RestrictToRepacking())

    mmf = pyrosetta.rosetta.core.select.movemap.MoveMapFactory()
    mmf.all_bb(setting=True)
    mmf.all_bondangles(setting=True)
    mmf.all_bondlengths(setting=True)
    mmf.all_chi(setting=True)
    mmf.all_jumps(setting=True)
    mmf.set_cartesian(setting=True)

    fr = pyrosetta.rosetta.protocols.relax.FastRelax(
        scorefxn_in=scorefxn, standard_repeats=repeats
    )
    fr.cartesian(True)
    fr.set_task_factory(tf)
    fr.set_movemap_factory(mmf)
    fr.min_type("lbfgs_armijo_nonmonotone")
    fr.apply(pose)


def fast_relax_with_coord_cst(pose, scorefxn, repeats: int = 1) -> None:
    """
    Cartesian FastRelax with coordinate constraints to the starting pose.

    Adds harmonic restraints anchoring every CA atom to its initial position.
    Suitable for refining homology models or AlphaFold predictions without
    drifting away from the reference conformation. Constraints are added
    via FastRelax's ``constrain_relax_to_start_coords`` flag, which uses the
    standard Rosetta ``coordinate_constraint`` term (weight inherited from
    ``ref2015_cart``).
    """
    tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.RestrictToRepacking())

    mmf = pyrosetta.rosetta.core.select.movemap.MoveMapFactory()
    mmf.all_bb(setting=True)
    mmf.all_bondangles(setting=True)
    mmf.all_bondlengths(setting=True)
    mmf.all_chi(setting=True)
    mmf.all_jumps(setting=True)
    mmf.set_cartesian(setting=True)

    fr = pyrosetta.rosetta.protocols.relax.FastRelax(
        scorefxn_in=scorefxn, standard_repeats=repeats
    )
    fr.cartesian(True)
    fr.constrain_relax_to_start_coords(True)
    fr.coord_constrain_sidechains(False)
    fr.ramp_down_constraints(True)
    fr.set_task_factory(tf)
    fr.set_movemap_factory(mmf)
    fr.min_type("lbfgs_armijo_nonmonotone")
    fr.apply(pose)


def _local_cartesian_relax(
    pose,
    scorefxn,
    focus_pose_idx: int,
    radius: float = 8.0,
    repeats: int = 1,
) -> None:
    """
    Local Cartesian FastRelax restricted to the neighbourhood of one residue.

    Used by the Cartesian ddG protocol (Frenz et al. 2020) to relax the
    mutation site without paying for a full-protein relax. Backbone and
    side-chain DOFs are enabled only inside a sphere of *radius* Angstrom
    around the focus residue; everything else is fixed in place.

    Parameters
    ----------
    pose : pyrosetta.Pose
        Pose to relax in place.
    scorefxn : pyrosetta.ScoreFunction
        Cartesian-compatible score function (``ref2015_cart``).
    focus_pose_idx : int
        1-based Pose index of the focus residue (the mutation site).
    radius : float
        Neighbourhood radius in Angstrom. Default 8.0 follows Frenz 2020.
    repeats : int
        Number of FastRelax rounds. Default 1.
    """
    sel = pyrosetta.rosetta.core.select.residue_selector
    op  = pyrosetta.rosetta.core.pack.task.operation
    mm  = pyrosetta.rosetta.core.select.movemap

    focus = sel.ResidueIndexSelector()
    focus.set_index(focus_pose_idx)

    nbr = sel.NeighborhoodResidueSelector()
    nbr.set_focus_selector(focus)
    nbr.set_distance(radius)
    nbr.set_include_focus_in_subset(True)

    not_nbr = sel.NotResidueSelector(nbr)

    tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
    tf.push_back(op.InitializeFromCommandline())
    tf.push_back(op.RestrictToRepacking())
    tf.push_back(op.OperateOnResidueSubset(op.PreventRepackingRLT(), not_nbr))

    try:
        enable = mm.move_map_action.mm_enable
    except AttributeError:
        enable = mm.mm_enable

    mmf = mm.MoveMapFactory()
    mmf.all_bb(setting=False)
    mmf.all_chi(setting=False)
    mmf.all_jumps(setting=False)
    mmf.add_bb_action(enable, nbr)
    mmf.add_chi_action(enable, nbr)
    mmf.set_cartesian(setting=True)

    fr = pyrosetta.rosetta.protocols.relax.FastRelax(
        scorefxn_in=scorefxn, standard_repeats=repeats
    )
    fr.cartesian(True)
    fr.set_task_factory(tf)
    fr.set_movemap_factory(mmf)
    fr.min_type("lbfgs_armijo_nonmonotone")
    fr.apply(pose)


def _pre_relax_decoy_worker(args: tuple) -> tuple:
    """
    Subprocess worker that produces a single relaxed decoy.

    Each decoy must run in its own process because ``pyrosetta.init`` is
    honoured only on first call within a process; running multiple decoys
    in-process would cause every decoy to share the same seed regardless
    of the ``-jran`` flag passed on subsequent ``init`` calls.
    """
    pdb_in, decoy_idx, seed, decoy_path = args

    pyrosetta.init(extra_options=(
        f"-mute all -constant_seed -jran {seed} "
        "-ex1 -ex2 -use_input_sc -flip_HNQ"
    ))
    scorefxn = _build_scorefxn_with_hbond()

    pose = pose_from_pdb(pdb_in)
    fast_relax_with_coord_cst(pose, scorefxn, repeats=1)
    score = scorefxn(pose)
    pose.dump_pdb(decoy_path)

    return decoy_idx, seed, decoy_path, score


def pre_relax_multi_decoy(
    pdb_in: str,
    output_dir: str,
    n_decoys: int = 5,
    base_seed: int = 12345,
    n_cpu: int = None,
) -> tuple:
    """
    Generate N relaxed decoys with coordinate constraints, return the lowest-energy one.

    Standard Rosetta preparation for homology / AlphaFold models (Conway 2014,
    Nivon 2013): N independent Cartesian FastRelax runs with harmonic
    restraints to the starting CA coordinates, producing a small ensemble.
    The decoy with the lowest ``ref2015_cart`` total score is selected as
    the input for downstream DMS.

    Each decoy runs in its own subprocess so that ``-jran <seed>`` is honoured
    independently. Decoys are produced in parallel up to ``n_cpu`` workers.

    Parameters
    ----------
    pdb_in : str
        Path to the input PDB file (the model to refine).
    output_dir : str
        Directory where decoy PDBs and the energy log will be written.
    n_decoys : int
        Number of independent FastRelax runs.
    base_seed : int
        First random seed; decoy *i* uses ``base_seed + i``.
    n_cpu : int or None
        Pool size. ``None`` -> ``min(n_decoys, os.cpu_count())``.

    Returns
    -------
    tuple[str, float, list[float]]
        ``(best_pdb_path, best_total_score, all_total_scores)``.
    """
    from pathlib import Path
    import shutil

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if n_cpu is None:
        n_cpu = min(n_decoys, os.cpu_count() or 1)

    worker_args = [
        (pdb_in, i, base_seed + i, str(out / f"decoy_{i:03d}.pdb"))
        for i in range(n_decoys)
    ]

    with multiprocessing.Pool(processes=n_cpu) as pool:
        results = pool.map(_pre_relax_decoy_worker, worker_args)

    results.sort(key=lambda r: r[0])
    decoy_paths = [r[2] for r in results]
    decoy_scores = [r[3] for r in results]
    decoy_seeds = [r[1] for r in results]

    for i, score in enumerate(decoy_scores):
        log.info("Decoy %d/%d (seed %d): total_score = %.3f REU",
                 i + 1, n_decoys, decoy_seeds[i], score)

    best_idx = min(range(n_decoys), key=lambda i: decoy_scores[i])
    best_pdb = out / "minimum_energy.pdb"
    shutil.copyfile(decoy_paths[best_idx], str(best_pdb))

    log_df = pd.DataFrame({
        "decoy": list(range(n_decoys)),
        "seed": decoy_seeds,
        "total_score": decoy_scores,
        "selected": [i == best_idx for i in range(n_decoys)],
    })
    log_df.to_csv(out / "decoy_energies.csv", index=False)

    log.info(
        "Selected decoy %d (seed %d, E = %.3f REU) -> %s",
        best_idx, decoy_seeds[best_idx], decoy_scores[best_idx], best_pdb,
    )
    return str(best_pdb), decoy_scores[best_idx], decoy_scores


def _dms_cartesian_ddg_worker(args: tuple) -> str:
    """
    Cartesian ddG worker (Frenz et al. 2020) for one residue position.

    For the focus position:

      1. Relax the WT pose locally (8 A around the focus) ``n_replicas``
         times, scoring each replica.
      2. For each of the 20 canonical amino acids:
         a. Apply the mutation via ``mutate_repack``.
         b. Run local Cartesian FastRelax ``n_replicas`` times on the mutant.
         c. ddG = mean(E_mut) - mean(E_wt). Per-term ddG values are computed
            from per-residue energies summed over the entire pose, again as
            mean over replicas.

    The WT relax loop runs only once per position (not per amino acid),
    which is the standard optimisation.

    args = (pdb_path, pose_index, pdb_index, chain, save_structures,
            output_dir, n_replicas, neighborhood_radius, base_seed)
    """
    from pathlib import Path

    # use_relax is optional (10th element); defaults True for backward compat
    use_relax = args[9] if len(args) > 9 else True
    (pdb_path, pose_index, pdb_index, chain, save_structures,
     output_dir, n_replicas, neighborhood_radius, base_seed) = args[:9]

    seed = base_seed + pose_index
    pyrosetta.init(options=(
        f"-mute all -constant_seed -jran {seed} "
        "-ex1 -ex2 -use_input_sc -flip_HNQ"
    ))
    scorefxn = _build_scorefxn_with_hbond()

    pose_wt = pose_from_pdb(pdb_path)
    scorefxn.score(pose_wt)

    wt_res_name = pose_wt.residue(pose_index).name()[:3]
    wt_aa1 = _AA_3TO1.get(wt_res_name, "X")

    pdb_dir = Path(output_dir) / "structures"
    csv_dir = Path(output_dir) / "csv"
    if save_structures:
        pdb_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    wt_replica_terms = []
    wt_replica_total = []
    for r in range(n_replicas):
        wt_p = pose_wt.clone()
        if use_relax:
            _local_cartesian_relax(
                wt_p, scorefxn, pose_index,
                radius=neighborhood_radius, repeats=1,
            )
        wt_replica_total.append(scorefxn(wt_p))
        wt_replica_terms.append(Get_energy_per_term(wt_p, scorefxn))

    energy_terms = list(wt_replica_terms[0].keys())
    wt_mean_total = sum(wt_replica_total) / n_replicas
    wt_mean_terms = {
        t: sum(rep[t] for rep in wt_replica_terms) / n_replicas
        for t in energy_terms
    }

    rows = []
    for aa in _AA_20:
        mut_replica_total = []
        mut_replica_terms = []
        for r in range(n_replicas):
            mut_p = mutate_repack(pose_wt, pose_index, aa, scorefxn)
            if use_relax:
                _local_cartesian_relax(
                    mut_p, scorefxn, pose_index,
                    radius=neighborhood_radius, repeats=1,
                )
            mut_replica_total.append(scorefxn(mut_p))
            mut_replica_terms.append(Get_energy_per_term(mut_p, scorefxn))

            if save_structures and r == 0:
                out_pdb = pdb_dir / f"{chain}{pdb_index}_{wt_aa1}{aa}.pdb"
                mut_p.dump_pdb(str(out_pdb))

        mut_mean_total = sum(mut_replica_total) / n_replicas
        mut_mean_terms = {
            t: sum(rep[t] for rep in mut_replica_terms) / n_replicas
            for t in energy_terms
        }

        summary = {
            "Position_Pose":   pose_index,
            "Position_PDB":    pdb_index,
            "Chain":           chain,
            "WT":              wt_aa1,
            "Mutation":        aa,
            "Label":           f"{wt_aa1}{pdb_index}{aa}",
            "ddG_total_score": mut_mean_total - wt_mean_total,
            "wt_total_score":  wt_mean_total,
            "mut_total_score": mut_mean_total,
            "n_replicas":      n_replicas,
        }
        for t in energy_terms:
            summary[f"ddG_{t}"] = mut_mean_terms[t] - wt_mean_terms[t]
        rows.append(summary)

    df_out = pd.DataFrame(rows)
    csv_path = csv_dir / f"DMS_pos{chain}{pdb_index}.csv"
    df_out.to_csv(csv_path, index=False)
    log.info("Position %s%d done -> %s", chain, pdb_index, csv_path)
    return str(csv_path)


def Run_DMS_CartesianDDG_Parallel(
    pdb: str,
    n_cpu: int,
    positions: list = None,
    chain: str = None,
    save_structures: bool = False,
    output_dir: str = "./DMS_output",
    n_replicas: int = 3,
    neighborhood_radius: float = 8.0,
    base_seed: int = 12345,
) -> pd.DataFrame:
    """
    Parallel DMS using the Cartesian ddG protocol (Frenz et al. 2020).

    For every requested position, ddG is estimated as

        ddG = mean_r [ E(mut_r) ] - mean_r [ E(wt_r) ]

    where each replica r runs an independent local Cartesian FastRelax in
    a sphere of *neighborhood_radius* around the focus residue. The WT
    replicas are computed once per position; the 20 mutants reuse the
    same WT mean (this matches the published protocol and avoids 20x
    redundant WT relaxes).

    Parameters mirror :func:`Run_DMS_Parallel`. *fast_relax_repeats* is
    replaced by *n_replicas* and *neighborhood_radius*.
    """
    from pathlib import Path

    if (positions is None) != (chain is None):
        raise ValueError(
            "Both 'positions' and 'chain' must be provided together, "
            "or both omitted to scan the entire structure."
        )

    pyrosetta.init(options="-mute all -constant_seed -jran 1")
    _pose_tmp = pose_from_pdb(pdb)
    index_df = PDB_pose_dictionairy(_pose_tmp)

    if positions is None:
        resolved = [
            (int(row["IndexPose"]), int(row["IndexPDB"]), row["Chain"])
            for _, row in index_df.iterrows()
        ]
        log.info(
            "No positions specified - scanning all %d residues in %s",
            len(resolved), pdb,
        )
    else:
        chain_df = index_df[index_df["Chain"] == chain]
        resolved = []
        missing = []
        for pdb_idx in positions:
            row = chain_df[chain_df["IndexPDB"] == pdb_idx]
            if row.empty:
                missing.append(pdb_idx)
            else:
                resolved.append((int(row["IndexPose"].iloc[0]), pdb_idx, chain))
        if missing:
            raise ValueError(
                f"PDB residue numbers not found in chain '{chain}': {missing}"
            )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    csv_dir = Path(output_dir) / "csv"

    pending = []
    skipped = []
    for pose_idx, pdb_idx, ch in resolved:
        existing = csv_dir / f"DMS_pos{ch}{pdb_idx}.csv"
        if existing.exists():
            skipped.append(str(existing))
        else:
            pending.append((pose_idx, pdb_idx, ch))

    if skipped:
        log.info("Resuming: %d positions already done, skipping.", len(skipped))

    log.info(
        "Cartesian ddG DMS | PDB: %s | Positions: %d (pending: %d) | CPUs: %d | "
        "Replicas: %d | Radius: %.1f A",
        pdb, len(resolved), len(pending), n_cpu, n_replicas, neighborhood_radius,
    )

    existing_csvs = skipped
    _POSITION_TIMEOUT = 1200  # 20 min per position before retry without relax

    if pending:
        worker_args = [
            (pdb, pose_idx, pdb_idx, ch, save_structures,
             output_dir, n_replicas, neighborhood_radius, base_seed, True)
            for pose_idx, pdb_idx, ch in pending
        ]
        new_csvs = []
        timed_out = []
        n_pending = len(worker_args)
        done_count = 0

        with multiprocessing.Pool(processes=n_cpu) as pool:
            async_results = [
                (arg, pool.apply_async(_dms_cartesian_ddg_worker, (arg,)))
                for arg in worker_args
            ]
            for arg, ar in async_results:
                ch_i, pdb_idx_i = arg[3], arg[2]
                try:
                    new_csvs.append(ar.get(timeout=_POSITION_TIMEOUT))
                    done_count += 1
                    log.info(
                        "Progress: %d/%d (%.0f%%) | last: %s%d",
                        done_count, n_pending,
                        100 * done_count / n_pending,
                        ch_i, pdb_idx_i,
                    )
                except multiprocessing.TimeoutError:
                    done_count += 1
                    log.warning(
                        "Position %s%d timed out after %ds - retrying without FastRelax "
                        "[%d/%d, %.0f%%]",
                        ch_i, pdb_idx_i, _POSITION_TIMEOUT,
                        done_count, n_pending,
                        100 * done_count / n_pending,
                    )
                    timed_out.append(arg)
                except Exception as exc:
                    done_count += 1
                    log.error("Position %s%d failed: %s", ch_i, pdb_idx_i, exc)

        if timed_out:
            log.info("Retrying %d timed-out positions without FastRelax", len(timed_out))
            retry_args = [(*arg[:9], False) for arg in timed_out]
            with multiprocessing.Pool(processes=n_cpu) as pool:
                new_csvs.extend(pool.map(_dms_cartesian_ddg_worker, retry_args))
    else:
        new_csvs = []

    csv_paths = existing_csvs + new_csvs

    dfs = [pd.read_csv(p) for p in csv_paths]
    df_all = pd.concat(dfs, ignore_index=True)

    report_path = Path(output_dir) / "DMS_report.csv"
    df_all.to_csv(report_path, index=False)
    log.info("Consolidated report saved -> %s", report_path)

    return df_all


def _dms_worker(args: tuple) -> str:
    """
    Subprocess worker that performs a full DMS scan for one residue position.
 
    This function is designed to run inside a ``multiprocessing.Pool`` worker.
    PyRosetta is reinitialised inside each subprocess because initialisation
    state is not shared across process boundaries.
 
    The worker computes **ΔΔG per energy term** for every one of the 20
    canonical amino acids at the target position, defined as:
 
        ΔΔG_term = Σ_residues [ E_term(mutant) − E_term(WT) ]
 
    One summary row per mutation is written (summed over all residues), and
    results are saved to a per-position CSV file.
 
    Parameters
    ----------
    args : tuple
        A six-element tuple ``(pdb_path, pose_index, pdb_index, chain,
        save_structures, output_dir, fast_relax_repeats)`` where:
 
        - *pdb_path* (*str*): path to the wild-type PDB file.
        - *pose_index* (*int*): Rosetta Pose index (1-based) of the residue,
          pre-resolved by :func:`Run_DMS_Parallel` from the user-supplied PDB
          index and chain.
        - *pdb_index* (*int*): original PDB residue number, carried along for
          labelling purposes.
        - *chain* (*str*): chain identifier (e.g. ``'A'``), carried along for
          labelling purposes.
        - *save_structures* (*bool*): if ``True``, each mutant pose is dumped
          to a PDB file under ``<output_dir>/structures/``.
        - *output_dir* (*str*): root directory for all outputs.
        - *fast_relax_repeats* (*int*): number of FastRelax rounds applied
          after each mutation (0 = disabled).
 
    Returns
    -------
    str
        Absolute path to the CSV file written for this position
        (``<output_dir>/csv/DMS_pos<chain><pdb_index>.csv``).
 
    Side Effects
    ------------
    - Writes ``<output_dir>/csv/DMS_pos<chain><pdb_index>.csv``.
    - If *save_structures* is ``True``, writes one PDB per mutant to
      ``<output_dir>/structures/<chain><pdb_index>_<WT><MUT>.pdb``.
    """
    from pathlib import Path
 
    pdb_path, pose_index, pdb_index, chain, save_structures, output_dir, fast_relax_repeats = args
 
    pyrosetta.init(options="-mute all")
    scorefxn = _build_scorefxn_with_hbond()
 
    pose_wt = pose_from_pdb(pdb_path)
    scorefxn.score(pose_wt)
 
    wt_energy_df = Energy_contribution_DMS(pose_wt, by_term=True)
    wt_res_name  = pose_wt.residue(pose_index).name()[:3]
    wt_aa1       = _AA_3TO1.get(wt_res_name, "X")
 
    pdb_dir = Path(output_dir) / "structures"
    csv_dir = Path(output_dir) / "csv"
    if save_structures:
        pdb_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
 
    # Identify energy-term columns (all columns that are not metadata)
    meta_cols = {"Residue_Index_Pose", "Residue_Index_PDB",
                 "Residue_Name", "Residue_Name1", "Chain"}
    energy_cols = [c for c in wt_energy_df.columns if c not in meta_cols]
 
    rows = []
    for aa in _AA_20:
        mut_pose = mutate_repack(pose_wt, pose_index, aa, scorefxn)
 
        if fast_relax_repeats > 0:
            log.debug("  FastRelax %s%s%s→%s (repeats=%d)",
                      wt_aa1, pdb_index, aa, aa, fast_relax_repeats)
            fast_relax(mut_pose, scorefxn, repeats=fast_relax_repeats)
 
        mut_energy_df = Energy_contribution_DMS(mut_pose, by_term=True)
 
        # ΔΔG per term = sum over all residues of (E_mut - E_wt)
        ddg_series = (
            mut_energy_df[energy_cols].subtract(wt_energy_df[energy_cols])
        ).sum()
 
        # Total score of each state = sum of all weighted per-residue terms
        wt_total  = scorefxn(pose_wt)
        #wt_total  = wt_energy_df[energy_cols].values.sum()
        #mut_total = mut_energy_df[energy_cols].values.sum()
        mut_total = scorefxn(mut_pose)
 
        summary = {
            "Position_Pose":   pose_index,
            "Position_PDB":    pdb_index,
            "Chain":           chain,
            "WT":              wt_aa1,
            "Mutation":        aa,
            "Label":           f"{wt_aa1}{pdb_index}{aa}",
            "ddG_total_score": mut_total - wt_total,
        }
        for col in energy_cols:
            summary[f"ddG_{col}"] = ddg_series[col]
 
        rows.append(summary)
 
        if save_structures:
            out_pdb = pdb_dir / f"{chain}{pdb_index}_{wt_aa1}{aa}.pdb"
            mut_pose.dump_pdb(str(out_pdb))
            log.info("  saved %s", out_pdb.name)
 
    df_out = pd.DataFrame(rows)
    csv_path = csv_dir / f"DMS_pos{chain}{pdb_index}.csv"
    df_out.to_csv(csv_path, index=False)
    log.info("Position %s%d done → %s", chain, pdb_index, csv_path)
    return str(csv_path)


def Run_DMS_Parallel(
    pdb: str,
    n_cpu: int,
    positions: list = None,
    chain: str = None,
    save_structures: bool = False,
    output_dir: str = "./DMS_output",
    fast_relax_repeats: int = 0,
) -> pd.DataFrame:
    """
    Run in silico DMS for multiple residue positions in parallel.
 
    By default, scans **every residue in the entire structure**. Optionally,
    the scan can be restricted to a specific set of residues by providing
    *positions* (in PDB numbering) and *chain* together.
 
    Accepts residue positions in **PDB numbering** (as they appear in the PDB
    file). The conversion to Rosetta's internal Pose numbering is handled
    automatically in the main process before any workers are spawned, so
    callers never need to know about Pose indices.
 
    Distributes one :func:`_dms_worker` call per position across a
    ``multiprocessing.Pool``.  Each worker is fully independent: it
    reinitialises PyRosetta, scores the wild-type pose, iterates over all
    20 amino acids, and writes its own CSV.  After all workers complete,
    the per-position CSVs are concatenated into a single consolidated report
    (``<output_dir>/DMS_report.csv``) and returned as a DataFrame.
 
    The reported metric for each mutation is **ΔΔG per energy term**
    (summed over all residues of the structure):
 
        ΔΔG_term = Σ_residues [ E_term(mutant) − E_term(WT) ]
 
    Parameters
    ----------
    pdb : str
        Path to the wild-type PDB file.
    n_cpu : int
        Number of parallel worker processes.
    positions : list[int] or None, optional
        Residue numbers **in PDB numbering** to scan (as written in the PDB
        file, e.g. ``[10, 11, 12, 45, 46]``). Must be provided together with
        *chain*. If ``None`` (default), all residues in the structure are
        scanned regardless of chain.
    chain : str or None, optional
        Single-letter chain identifier that all *positions* belong to
        (e.g. ``'A'``). Required when *positions* is provided. If ``None``
        (default), all chains are scanned. To scan residues across multiple
        specific chains, call this function once per chain and concatenate
        the results.
    save_structures : bool, optional
        If ``True``, each mutant structure is saved as a PDB file under
        ``<output_dir>/structures/``. Default: ``False``.
    output_dir : str, optional
        Root directory for all output files. Sub-directories ``csv/`` and
        (if *save_structures*) ``structures/`` are created automatically.
        Default: ``"./DMS_output"``.
    fast_relax_repeats : int, optional
        Number of FastRelax rounds applied after each mutation via
        :func:`fast_relax`. ``0`` disables relaxation entirely (default).
        Use ``1`` for a light relaxation pass; higher values improve
        geometry but scale runtime linearly.
 
    Returns
    -------
    pd.DataFrame
        Consolidated DataFrame with one row per (position × mutation).
        Columns include ``Position_Pose``, ``Position_PDB``, ``Chain``,
        ``WT``, ``Mutation``, ``Label``, ``ddG_total_score``, and one
        ``ddG_<term>`` column per active energy term.
 
    Raises
    ------
    ValueError
        If *positions* is provided without *chain*, or vice versa.
        If any PDB residue number in *positions* is not found in *chain*.
 
    Side Effects
    ------------
    - Writes per-position CSVs to ``<output_dir>/csv/DMS_pos<chain><N>.csv``.
    - Writes the consolidated report to ``<output_dir>/DMS_report.csv``.
    - If *save_structures* is ``True``, writes mutant PDB files to
      ``<output_dir>/structures/``.
 
    Notes
    -----
    ``multiprocessing.set_start_method("spawn")`` is recommended on macOS
    and Windows to avoid issues with forked processes and PyRosetta's
    internal state. Call it once in your ``if __name__ == "__main__":``
    block before invoking this function.
 
    Examples
    --------
    Scan the entire structure (default):
 
    >>> df = Run_DMS_Parallel(pdb="complex.pdb", n_cpu=8)
 
    Scan specific residues on chain A:
 
    >>> df = Run_DMS_Parallel(
    ...     pdb="complex.pdb",
    ...     n_cpu=8,
    ...     positions=[10, 11, 12, 45, 46],
    ...     chain="A",
    ... )
    """
    from pathlib import Path
 
    # ------------------------------------------------------------------
    # Validate positions / chain arguments
    # ------------------------------------------------------------------
    if (positions is None) != (chain is None):
        raise ValueError(
            "Both 'positions' and 'chain' must be provided together, "
            "or both must be omitted to scan the entire structure."
        )
 
    # ------------------------------------------------------------------
    # Load structure and build PDB ↔ Pose index map
    # ------------------------------------------------------------------
    pyrosetta.init(options="-mute all")
    _pose_tmp = pose_from_pdb(pdb)
    index_df  = PDB_pose_dictionairy(_pose_tmp)
 
    # ------------------------------------------------------------------
    # Resolve target positions → list of (pose_index, pdb_index, chain)
    # ------------------------------------------------------------------
    if positions is None:
        # Default: scan every residue in the structure, all chains
        resolved = [
            (int(row["IndexPose"]), int(row["IndexPDB"]), row["Chain"])
            for _, row in index_df.iterrows()
        ]
        log.info(
            "No positions specified — scanning all %d residues in %s",
            len(resolved), pdb,
        )
    else:
        # Restricted scan: validate and resolve PDB indices for the given chain
        chain_df = index_df[index_df["Chain"] == chain]
        resolved = []
        missing  = []
        for pdb_idx in positions:
            row = chain_df[chain_df["IndexPDB"] == pdb_idx]
            if row.empty:
                missing.append(pdb_idx)
            else:
                resolved.append((int(row["IndexPose"].iloc[0]), pdb_idx, chain))
 
        if missing:
            raise ValueError(
                f"The following PDB residue numbers were not found in chain "
                f"'{chain}': {missing}. Check that the numbers match the PDB "
                f"file and that the correct chain is specified."
            )
 
    log.info(
        "Starting DMS | PDB: %s | Positions to scan: %d | "
        "CPUs: %d | Save structures: %s | FastRelax repeats: %d",
        pdb, len(resolved), n_cpu, save_structures, fast_relax_repeats,
    )
 
    Path(output_dir).mkdir(parents=True, exist_ok=True)
 
    worker_args = [
        (pdb, pose_idx, pdb_idx, ch, save_structures, output_dir, fast_relax_repeats)
        for pose_idx, pdb_idx, ch in resolved
    ]
 
    with multiprocessing.Pool(processes=n_cpu) as pool:
        csv_paths = pool.map(_dms_worker, worker_args)
 
    dfs = [pd.read_csv(p) for p in csv_paths]
    df_all = pd.concat(dfs, ignore_index=True)
 
    report_path = Path(output_dir) / "DMS_report.csv"
    df_all.to_csv(report_path, index=False)
    log.info("Consolidated report saved → %s", report_path)
 
    return df_all
# ===========================================================================
# SECTION 8b — DMS Visualisation
# ===========================================================================
def plot_dms_heatmap(
    df: "pd.DataFrame",
    metric: str = "ddG_total_score",
    output_path: str = None,
    title: str = None,
    cmap: str = "RdBu_r",
    center: float = 0.0,
    figsize: tuple = None,
) -> None:
    """
    Generate a publication-quality heatmap from a DMS results DataFrame.
 
    Rows represent the 20 canonical amino acids (mutations); columns represent
    scanned residue positions, labelled as <WT><PDB_number> (e.g. A42).
    Cell colour encodes the chosen *metric* (default: ddG_total_score).
 
    The wild-type amino acid at each position is marked with a black dot so
    it is immediately identifiable on the plot.
 
    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:. Must contain columns Label,
        WT, Position_PDB, Mutation, and the column specified by
        *metric*.
    metric : str, optional
        Name of the column to use as the colour value.
        Default: "ddG_total_score".
    output_path : str or None, optional
        If provided, the figure is saved to this path (e.g.
        "./figures/dms_heatmap.png"). The file format is inferred from
        the extension. If None (default), the figure is displayed
        interactively.
    title : str or None, optional
        Title shown above the heatmap. Defaults to
        "DMS Heatmap — <metric>".
    cmap : str, optional
        Matplotlib / seaborn colormap name. Default: "RdBu_r"
        (blue = stabilising, red = destabilising).
    center : float, optional
        Value at which the colormap is centred. Default: 0.0.
    figsize : tuple or None, optional
        Figure size (width, height) in inches. If None, the size is
        inferred automatically from the number of positions and amino acids.
 
    Returns
    -------
    None
        The figure is either saved to *output_path* or shown interactively.
 
    Raises
    ------
    ImportError
        If matplotlib or seaborn are not installed.
    KeyError
        If *metric* is not a column in *df*.
 
    Example
    -------
    >>> df = Run_DMS_Parallel(pdb="complex.pdb", positions=[10,11,12], chain="A", n_cpu=4)
    >>> plot_dms_heatmap(df, metric="ddG_total_score", output_path="./dms_heatmap.png")
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import seaborn as sns
    except ImportError as exc:
        raise ImportError(
            "plot_dms_heatmap requires matplotlib and seaborn. "
            "Install them with: pip install matplotlib seaborn"
        ) from exc
 
    if metric not in df.columns:
        raise KeyError(
            f"Column '{metric}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )
 
    # ------------------------------------------------------------------
    # Build pivot table: rows = mutations, columns = positions
    # ------------------------------------------------------------------
    # Build position label: WT + PDB number  (e.g. "A42")
    df = df.copy()
    df["pos_label"] = df["WT"] + df["Position_PDB"].astype(str)
 
    # Sort columns by PDB position number
    position_order = (
        df[["Position_PDB", "pos_label"]]
        .drop_duplicates()
        .sort_values("Position_PDB")["pos_label"]
        .tolist()
    )
 
    # Sort rows by standard biochemistry amino acid order
    aa_order = ["G", "A", "V", "L", "I", "P", "F", "W", "M", "S",
                "T", "C", "Y", "H", "D", "E", "N", "Q", "K", "R"]
 
    pivot = df.pivot_table(
        index="Mutation",
        columns="pos_label",
        values=metric,
        aggfunc="first",
    )
    pivot = pivot.reindex(index=aa_order, columns=position_order)
 
    # ------------------------------------------------------------------
    # Build wild-type mask: True where Mutation == WT at that position
    # ------------------------------------------------------------------
    wt_map = (
        df[["pos_label", "WT"]]
        .drop_duplicates()
        .set_index("pos_label")["WT"]
        .to_dict()
    )
    wt_mask = pd.DataFrame(False, index=pivot.index, columns=pivot.columns)
    for pos_lbl, wt_aa in wt_map.items():
        if pos_lbl in wt_mask.columns and wt_aa in wt_mask.index:
            wt_mask.loc[wt_aa, pos_lbl] = True
 
    # ------------------------------------------------------------------
    # Figure geometry
    # ------------------------------------------------------------------
    n_pos = len(position_order)
    n_aa  = len(aa_order)
 
    if figsize is None:
        w = max(8, n_pos * 0.7 + 2)
        h = max(5, n_aa  * 0.4 + 2)
        figsize = (w, h)
 
    fig, ax = plt.subplots(figsize=figsize)
 
    # ------------------------------------------------------------------
    # Draw heatmap
    # ------------------------------------------------------------------
    vmax = df[metric].abs().quantile(0.97)   # robust colour scale
    vmin = -vmax
 
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=cmap,
        center=center,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.3,
        linecolor="#cccccc",
        cbar_kws={"label": metric, "shrink": 0.7},
        annot=False,
    )
 
    # Mark wild-type positions with a black dot
    for row_i, aa in enumerate(pivot.index):
        for col_j, pos in enumerate(pivot.columns):
            if wt_mask.loc[aa, pos]:
                ax.plot(
                    col_j + 0.5, row_i + 0.5,
                    marker="o", color="black", markersize=5, zorder=5,
                )
 
    # ------------------------------------------------------------------
    # Labels and aesthetics
    # ------------------------------------------------------------------
    ax.set_xlabel("Position", fontsize=11, labelpad=8)
    ax.set_ylabel("Mutation", fontsize=11, labelpad=8)
    ax.set_title(title or f"DMS Heatmap — {metric}", fontsize=13, pad=12)
 
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)
 
    # Legend for wild-type marker
    wt_dot = mpatches.Patch(color="black", label="Wild-type")
    ax.legend(
        handles=[wt_dot], loc="upper left",
        bbox_to_anchor=(1.12, 1.0), frameon=False, fontsize=9,
    )
 
    plt.tight_layout()
 
    if output_path:
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        log.info("Heatmap saved → %s", output_path)
    else:
        plt.show()
 
    plt.close(fig)
 


# ===========================================================================
# SECTION 9 — Sequence Modeling
# ===========================================================================

def read_pose(pdb):
    """
    Initialise PyRosetta, load a PDB file, and create a ``ref2015_cart`` score function.

    This is a convenience entry point for sequence modeling workflows. It
    initialises the PyRosetta framework (if not already done), reads the
    structure from disk, creates the score function, and scores the pose once
    to ensure that the internal energy graph is fully populated before any
    downstream operations.

    Parameters
    ----------
    pdb : str
        Path to the PDB file to load.

    Returns
    -------
    tuple[pyrosetta.Pose, pyrosetta.ScoreFunction]
        - **pose**: Rosetta Pose object loaded from *pdb*.
        - **scorefxn**: ``ref2015_cart`` score function that has already been
          applied to the pose.

    Notes
    -----
    PyRosetta is initialised with default options. If custom flags are needed
    (e.g. ``-auto_setup_metals``), call ``pyrosetta.init()`` manually before
    invoking this function.
    """
    pyrosetta.init()
    pose = pose_from_pdb(pdb)
    scorefxn = pyrosetta.create_score_function("ref2015_cart.wts")
    scorefxn(pose)
    return pose, scorefxn


def Get_residues_from_pose(pose):
    """
    Extract the one-letter sequence and the corresponding Pose index list from a pose.

    Iterates over all residues in the pose in Pose-index order and builds both
    a sequence string and a parallel list of integer indices, which are needed
    by :func:`Compare_sequences` and :func:`model_sequence`.

    Parameters
    ----------
    pose : pyrosetta.Pose
        Pose from which to extract sequence information.

    Returns
    -------
    tuple[str, list[int]]
        - **sequence** (*str*): one-letter amino acid sequence of the full pose
          (all chains concatenated in Pose order).
        - **residue_numbers** (*list[int]*): Pose indices (1-based) for every
          residue, in the same order as *sequence*.

    Example
    -------
    >>> seq, idx = Get_residues_from_pose(pose)
    >>> print(seq[:5], idx[:5])
    MKTII [1, 2, 3, 4, 5]
    """
    residue_numbers = list(range(1, pose.size() + 1))
    sequence = "".join(pose.residue(r).name1() for r in residue_numbers)
    return sequence, residue_numbers


def Compare_sequences(before_seq, after_seq, indexes):
    """
    Compare a reference sequence against a target sequence and return the mutations.

    Performs a positional comparison between *before_seq* (current structure
    sequence) and *after_seq* (desired target sequence). Positions that differ
    are collected into a dictionary and printed to stdout with a brief summary.

    Parameters
    ----------
    before_seq : str
        One-letter sequence of the starting (wild-type or template) structure.
        Must have the same length as *after_seq*.
    after_seq : str
        One-letter sequence of the desired target structure.
    indexes : list[int]
        Rosetta Pose indices corresponding to each position in *before_seq* /
        *after_seq*. Typically obtained from :func:`Get_residues_from_pose`.

    Returns
    -------
    dict[int, str]
        Mapping of ``{pose_index: target_one_letter_code}`` for every position
        where *before_seq* and *after_seq* differ.  An empty dict is returned
        when the sequences are identical.

    Raises
    ------
    ValueError
        If *before_seq* and *after_seq* have different lengths.

    Example
    -------
    >>> mutations = Compare_sequences("MKTII", "MKTIA", [1, 2, 3, 4, 5])
    New mutation: I5A
    >>> print(mutations)
    {5: 'A'}
    """
    if len(before_seq) != len(after_seq):
        raise ValueError(
            f"Sequence length mismatch: before={len(before_seq)}, after={len(after_seq)}."
        )

    mutations = {}
    for idx, (wt_res, mut_res) in enumerate(zip(before_seq, after_seq)):
        if wt_res != mut_res:
            pose_idx = indexes[idx]
            mutations[pose_idx] = mut_res
            print(f"New mutation: {wt_res}{pose_idx}{mut_res}")
    return mutations


def model_sequence(pose, mutations, scorefxn, relax=True):
    """
    Apply a set of point mutations to a pose and optionally relax the result.

    Iterates over the *mutations* dictionary and calls :func:`mutate_repack`
    sequentially for each position. If *relax* is ``True``, a Cartesian
    FastRelax pass (via :func:`pack_relax`) is applied to the fully mutated
    pose before returning, allowing the backbone and side chains to accommodate
    all changes simultaneously.

    Parameters
    ----------
    pose : pyrosetta.Pose
        Template pose. Not modified — a clone is used internally by
        :func:`mutate_repack` at each step.
    mutations : dict[int, str]
        Dictionary mapping Pose index (int, 1-based) to the desired one-letter
        amino acid code. Typically produced by :func:`Compare_sequences`.
        Pass an empty dict to skip mutations and only relax (if *relax=True*).
    scorefxn : pyrosetta.ScoreFunction
        Score function used for rotamer packing and relaxation.
    relax : bool, optional
        If ``True`` (default), run :func:`pack_relax` on the final mutated pose.
        Set to ``False`` to skip relaxation when speed is more important than
        structural quality (e.g. during preliminary screening).

    Returns
    -------
    pyrosetta.Pose
        New pose carrying all requested mutations, optionally relaxed.
    """
    new_pose = pose.clone()
    for index, target_aa in mutations.items():
        new_pose = mutate_repack(
            starting_pose=new_pose,
            posi=index,
            amino=target_aa,
            scorefxn=scorefxn,
        )
    if relax:
        pack_relax(pose=new_pose, scorefxn=scorefxn)
    return new_pose


def Model_structure(pdb, sequence, output_name, relax=True):
    """
    Model a target amino acid sequence onto an existing backbone and write the result to disk.

    This is the primary high-level function for sequence modeling. Given a
    reference PDB structure and a target sequence, it:

    1. Loads the structure and score function via :func:`read_pose`.
    2. Extracts the current sequence and Pose indices via
       :func:`Get_residues_from_pose`.
    3. Identifies required mutations by comparing current vs. target sequences
       via :func:`Compare_sequences`.
    4. Applies all mutations and (optionally) relaxes the structure via
       :func:`model_sequence`.
    5. Dumps the final pose to ``<output_name>.pdb``.

    Parameters
    ----------
    pdb : str
        Path to the template PDB file (provides the backbone geometry).
    sequence : str
        Full one-letter target sequence. Must have the same length as the
        sequence extracted from *pdb* (all chains concatenated in Pose order).
    output_name : str
        Output file path **without** the ``.pdb`` extension. The modeled
        structure is written to ``<output_name>.pdb``.
    relax : bool, optional
        Passed directly to :func:`model_sequence`. If ``True`` (default), a
        FastRelax pass is applied after all mutations have been introduced.

    Returns
    -------
    pyrosetta.Pose
        The fully modeled (and optionally relaxed) pose. The same structure is
        also written to ``<output_name>.pdb``.

    Example
    -------
    >>> new_pose = Model_structure(
    ...     pdb="template.pdb",
    ...     sequence="ACDEFGHIKLMNPQRSTVWY",
    ...     output_name="./output/modeled_variant",
    ... )

    Notes
    -----
    Only residues that differ between the template and the target sequence are
    mutated; identical positions are left untouched, preserving their current
    rotamer conformation.
    """
    pose, scorefxn = read_pose(pdb)
    pose_init = pose.clone()

    current_seq, indexes = Get_residues_from_pose(pose=pose)
    mutations = Compare_sequences(
        before_seq=current_seq,
        after_seq=sequence,
        indexes=indexes,
    )
    new_pose = model_sequence(pose_init, mutations, scorefxn, relax)
    new_pose.dump_pdb(f"{output_name}.pdb")
    return new_pose


# ===========================================================================
# SECTION 10 — I/O Utilities
# ===========================================================================

def jd2_format(pdbfile, basename, outdir):
    """
    Convert a PDB file to Rosetta's JD2-compatible format.

    Loads the structure with ``beta_nov16`` corrections and PDB renumbering
    enabled, then writes a clean PDB file following the JD2 naming convention
    (``<basename>_jd2_0001.pdb``). Unrecognised residues are silently ignored.

    Parameters
    ----------
    pdbfile : str
        Path to the input PDB file.
    basename : str
        Base name for the output file (e.g. ``'my_protein'`` produces
        ``my_protein_jd2_0001.pdb``).
    outdir : str
        Directory where the converted file will be saved. Created
        automatically if it does not exist.
    """
    pyrosetta.init(extra_options=(
        "-corrections::beta_nov16 true "
        "-ignore_unrecognized_res "
        "-output_pose_energies_table false "
        "-renumber_pdb"
    ))
    pose = rosetta.core.import_pose.pose_from_file(pdbfile)
    os.makedirs(outdir, exist_ok=True)
    pose.dump_pdb(f"{outdir}/{basename}_jd2_0001.pdb")
