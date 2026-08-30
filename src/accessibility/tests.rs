use accesskit::{Action, Role, Toggled};
use test_r::test;

use super::*;

fn node_with_label<'a>(snapshot: &'a PeopleAccessibilitySnapshot, label: &str) -> &'a Node {
    snapshot
        .update
        .nodes
        .iter()
        .map(|(_, node)| node)
        .find(|node| node.label() == Some(label))
        .unwrap_or_else(|| panic!("missing accessibility node {label:?}"))
}

#[test]
fn overview_exposes_named_actionable_person_cards() {
    let snapshot = people_overview_snapshot(&[PersonAccessibilityItem {
        id: 42,
        name: "Ada Lovelace",
        confirmed: 7,
        suggested: 3,
    }]);
    let node = node_with_label(&snapshot, "Ada Lovelace, 7 confirmed faces, 3 suggestions");
    assert_eq!(node.role(), Role::Button);
    assert!(node.supports_action(Action::Click));
    assert!(!node.supports_action(Action::Focus));
    let (id, _) = snapshot
        .update
        .nodes
        .iter()
        .find(|(_, candidate)| std::ptr::eq(candidate, node))
        .unwrap();
    assert_eq!(
        snapshot.action_for(*id),
        Some(PeopleAccessibilityAction::OpenPerson(42))
    );
}

#[test]
fn face_tree_exposes_file_decision_selection_modes_and_actions() {
    let snapshot = people_faces_snapshot(
        &[FaceAccessibilityItem {
            face_id: 8,
            person_id: 42,
            file_name: "portrait.jpg",
            person_name: Some("Ada Lovelace"),
            decision: "suggested",
            selected: true,
            context: true,
            pending: false,
            can_accept: true,
            can_reject: true,
        }],
        false,
    );
    let tile = node_with_label(&snapshot, "portrait.jpg, Ada Lovelace, suggested");
    assert_eq!(tile.role(), Role::ListItem);
    assert_eq!(tile.is_selected(), Some(true));
    assert!(tile.supports_action(Action::Click));
    assert!(!tile.supports_action(Action::Focus));
    let toggle = node_with_label(&snapshot, "Show full context for portrait.jpg");
    assert_eq!(toggle.role(), Role::Switch);
    assert_eq!(toggle.toggled(), Some(Toggled::True));
    let accept = node_with_label(&snapshot, "Accept portrait.jpg as Ada Lovelace");
    assert_eq!(accept.role(), Role::Button);
    assert!(accept.supports_action(Action::Click));
    let reject = node_with_label(&snapshot, "Reject portrait.jpg for Ada Lovelace");
    assert_eq!(reject.role(), Role::Button);
    assert!(reject.supports_action(Action::Click));
}

#[test]
fn pending_face_actions_expose_busy_disabled_state_and_no_click() {
    let snapshot = people_faces_snapshot(
        &[FaceAccessibilityItem {
            face_id: 9,
            person_id: 43,
            file_name: "pending.raw",
            person_name: Some("Grace"),
            decision: "confirmed",
            selected: false,
            context: false,
            pending: true,
            can_accept: false,
            can_reject: true,
        }],
        true,
    );
    let tile = node_with_label(&snapshot, "pending.raw, Grace, confirmed");
    assert!(tile.is_busy());
    assert!(!tile.is_disabled());
    assert!(tile.supports_action(Action::Click));
    let context = node_with_label(&snapshot, "Show full context for pending.raw");
    assert!(!context.is_busy());
    assert!(!context.is_disabled());
    assert!(context.supports_action(Action::Click));
    let reject = node_with_label(&snapshot, "Reject pending.raw for Grace");
    assert!(reject.is_busy());
    assert!(reject.is_disabled());
    assert!(!reject.supports_action(Action::Click));
    let all = node_with_label(&snapshot, "Show full context for all face tiles");
    assert_eq!(all.toggled(), Some(Toggled::True));
}

#[test]
fn control_parity_exposes_role_state_and_exact_action() {
    let mut snapshot = people_overview_snapshot(&[]);
    snapshot.generation = 9;
    snapshot.append_controls(&[
        ControlAccessibilityItem {
            action: PeopleAccessibilityAction::Back,
            label: "Back to People".to_owned(),
            role: Role::Button,
            disabled: false,
            busy: false,
            toggled: None,
        },
        ControlAccessibilityItem {
            action: PeopleAccessibilityAction::ToggleAnalysisPause,
            label: "Pause automatic face analysis".to_owned(),
            role: Role::Switch,
            disabled: true,
            busy: true,
            toggled: Some(true),
        },
    ]);
    let back = node_with_label(&snapshot, "Back to People");
    assert_eq!(back.role(), Role::Button);
    assert!(back.supports_action(Action::Click));
    let pause = node_with_label(&snapshot, "Pause automatic face analysis");
    assert_eq!(pause.role(), Role::Switch);
    assert_eq!(pause.toggled(), Some(Toggled::True));
    assert!(pause.is_disabled());
    assert!(pause.is_busy());
    assert!(!pause.supports_action(Action::Click));
}

#[test]
fn queued_requests_keep_generation_and_deactivation_discards_them() {
    let bridge = PeopleAccessibilityBridge::new();
    let mut first = people_overview_snapshot(&[PersonAccessibilityItem {
        id: 42,
        name: "Ada",
        confirmed: 1,
        suggested: 0,
    }]);
    first.generation = 17;
    bridge.update(first, true);
    bridge.request_click(PeopleAccessibilityAction::OpenPerson(42));
    let mut second = people_overview_snapshot(&[PersonAccessibilityItem {
        id: 42,
        name: "Ada",
        confirmed: 2,
        suggested: 0,
    }]);
    second.generation = 18;
    bridge.update(second, true);
    assert_eq!(
        bridge.take_actions(),
        vec![(17, PeopleAccessibilityAction::OpenPerson(42))]
    );
    bridge.request_click(PeopleAccessibilityAction::OpenPerson(42));
    bridge.deactivate();
    assert!(bridge.take_actions().is_empty());
    #[cfg(target_os = "linux")]
    assert!(!bridge.adapter_active());
}
