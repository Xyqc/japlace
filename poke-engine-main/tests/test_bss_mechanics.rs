#![cfg(feature = "bss")]

use poke_engine::engine::abilities::Abilities;
use poke_engine::engine::generate_instructions::generate_instructions_for_bss_team_preview;
use poke_engine::engine::state::Weather;
use poke_engine::instruction::{
    BoostInstruction, ChangeWeather, Instruction, StateInstructions, SwitchInstruction,
};
use poke_engine::state::{PokemonBoostableStat, PokemonIndex, SideReference, State};

fn generate_team_preview_instructions_with_state_assertion(
    state: &mut State,
    side_one_move: (PokemonIndex, PokemonIndex, PokemonIndex),
    side_two_move: (PokemonIndex, PokemonIndex, PokemonIndex),
) -> Vec<StateInstructions> {
    let before_state_string = format!("{:?}", state);
    let instructions =
        generate_instructions_for_bss_team_preview(state, side_one_move, side_two_move);
    let after_state_string = format!("{:?}", state);
    assert_eq!(before_state_string, after_state_string);
    instructions
}

#[test]
fn test_bss_team_preview_get_all_options() {
    let state = State::default();
    let result = state.side_one.bss_team_preview_get_all_options();

    assert_eq!(60, result.len());
}

#[test]
fn test_bss_generate_team_preview_instructions() {
    let mut state = State::default();
    state.team_preview = true;

    let result = generate_team_preview_instructions_with_state_assertion(
        &mut state,
        (PokemonIndex::P0, PokemonIndex::P1, PokemonIndex::P2),
        (PokemonIndex::P0, PokemonIndex::P1, PokemonIndex::P2),
    );

    let expected_instructions = vec![StateInstructions {
        percentage: 100.0,
        instruction_list: vec![
            Instruction::Switch(SwitchInstruction {
                side_ref: SideReference::SideOne,
                previous_index: PokemonIndex::P0,
                next_index: PokemonIndex::P0,
            }),
            Instruction::Switch(SwitchInstruction {
                side_ref: SideReference::SideTwo,
                previous_index: PokemonIndex::P0,
                next_index: PokemonIndex::P0,
            }),
            Instruction::TeamPreviewFaintIndex(SideReference::SideOne, PokemonIndex::P3),
            Instruction::TeamPreviewFaintIndex(SideReference::SideOne, PokemonIndex::P4),
            Instruction::TeamPreviewFaintIndex(SideReference::SideOne, PokemonIndex::P5),
            Instruction::TeamPreviewFaintIndex(SideReference::SideTwo, PokemonIndex::P3),
            Instruction::TeamPreviewFaintIndex(SideReference::SideTwo, PokemonIndex::P4),
            Instruction::TeamPreviewFaintIndex(SideReference::SideTwo, PokemonIndex::P5),
            Instruction::ToggleTeamPreview,
        ],
    }];

    assert_eq!(result, expected_instructions);
}

#[test]
fn test_team_preview_with_abilities_active_in_order() {
    let mut state = State::default();
    state.side_one.pokemon.pkmn[0].ability = Abilities::SANDSTREAM;
    state.side_one.pokemon.pkmn[0].speed = 100;
    state.side_two.pokemon.pkmn[0].ability = Abilities::INTIMIDATE;
    state.side_two.pokemon.pkmn[0].speed = 150; // activates first
    state.team_preview = true;

    let result = generate_team_preview_instructions_with_state_assertion(
        &mut state,
        (PokemonIndex::P0, PokemonIndex::P1, PokemonIndex::P2),
        (PokemonIndex::P0, PokemonIndex::P1, PokemonIndex::P2),
    );

    let expected_instructions = vec![StateInstructions {
        percentage: 100.0,
        instruction_list: vec![
            Instruction::Switch(SwitchInstruction {
                side_ref: SideReference::SideOne,
                previous_index: PokemonIndex::P0,
                next_index: PokemonIndex::P0,
            }),
            Instruction::Switch(SwitchInstruction {
                side_ref: SideReference::SideTwo,
                previous_index: PokemonIndex::P0,
                next_index: PokemonIndex::P0,
            }),
            Instruction::Boost(BoostInstruction {
                side_ref: SideReference::SideOne,
                stat: PokemonBoostableStat::Attack,
                amount: -1,
            }),
            Instruction::ChangeWeather(ChangeWeather {
                new_weather: Weather::SAND,
                new_weather_turns_remaining: 5,
                previous_weather: Weather::NONE,
                previous_weather_turns_remaining: -1,
            }),
            Instruction::TeamPreviewFaintIndex(SideReference::SideOne, PokemonIndex::P3),
            Instruction::TeamPreviewFaintIndex(SideReference::SideOne, PokemonIndex::P4),
            Instruction::TeamPreviewFaintIndex(SideReference::SideOne, PokemonIndex::P5),
            Instruction::TeamPreviewFaintIndex(SideReference::SideTwo, PokemonIndex::P3),
            Instruction::TeamPreviewFaintIndex(SideReference::SideTwo, PokemonIndex::P4),
            Instruction::TeamPreviewFaintIndex(SideReference::SideTwo, PokemonIndex::P5),
            Instruction::ToggleTeamPreview,
        ],
    }];

    assert_eq!(result, expected_instructions);
}
