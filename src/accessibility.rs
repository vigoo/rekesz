use std::{
    collections::{HashMap, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
    path::PathBuf,
    sync::{Arc, Mutex},
};

use accesskit::{Action, ActionRequest, Node, NodeId, Role, Toggled, TreeId, TreeInfo, TreeUpdate};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum PeopleAccessibilityAction {
    OpenPerson(i64),
    SelectFace {
        face_id: i64,
        person_id: i64,
    },
    ToggleFaceContext {
        face_id: i64,
        person_id: i64,
    },
    AcceptFace {
        face_id: i64,
        person_id: i64,
    },
    RejectFace {
        face_id: i64,
        person_id: i64,
    },
    ToggleGridContext,
    Back,
    PreviousPage,
    NextPage,
    RenamePerson(i64),
    MergePerson(i64),
    UndoReview,
    OpenPhoto(PathBuf),
    MarkNotFace {
        face_id: i64,
        person_id: i64,
    },
    OpenReviewQueue,
    OpenUnnamedFaces,
    UseWholeLibrary(bool),
    CycleSort,
    ToggleAnalysisPause,
    RetryFailedAnalysis,
    RetryPage,
    SelectTarget(i64),
    BulkDecision {
        decision: crate::PeopleDecision,
        faces: Vec<(i64, i64)>,
    },
    CreatePerson,
    MoveSelected(i64),
    UseAsRepresentative,
    ClearSelection,
    CancelDialog,
    ConfirmDialog,
    PreviousTargetPage,
    NextTargetPage,
}

#[derive(Clone)]
pub struct ControlAccessibilityItem {
    pub action: PeopleAccessibilityAction,
    pub label: String,
    pub role: Role,
    pub disabled: bool,
    pub busy: bool,
    pub toggled: Option<bool>,
}

#[derive(Clone, Copy)]
pub struct PersonAccessibilityItem<'a> {
    pub id: i64,
    pub name: &'a str,
    pub confirmed: i64,
    pub suggested: i64,
}

#[derive(Clone, Copy)]
pub struct FaceAccessibilityItem<'a> {
    pub face_id: i64,
    pub person_id: i64,
    pub file_name: &'a str,
    pub person_name: Option<&'a str>,
    pub decision: &'a str,
    pub selected: bool,
    pub context: bool,
    pub pending: bool,
    pub can_accept: bool,
    pub can_reject: bool,
}

#[derive(Clone)]
pub struct PeopleAccessibilitySnapshot {
    pub update: TreeUpdate,
    pub generation: u64,
    actions: HashMap<NodeId, PeopleAccessibilityAction>,
}

impl PeopleAccessibilitySnapshot {
    pub fn action_for(&self, node: NodeId) -> Option<PeopleAccessibilityAction> {
        self.actions.get(&node).cloned()
    }

    pub fn append_controls(&mut self, controls: &[ControlAccessibilityItem]) {
        let mut ids = Vec::with_capacity(controls.len());
        for control in controls {
            let mut hasher = DefaultHasher::new();
            control.action.hash(&mut hasher);
            let id = NodeId(hasher.finish() | (1 << 62));
            let mut node = actionable(
                control.role,
                control.label.clone(),
                control.disabled,
                control.busy,
            );
            if let Some(toggled) = control.toggled {
                node.set_toggled(if toggled {
                    Toggled::True
                } else {
                    Toggled::False
                });
            }
            self.update.nodes.push((id, node));
            self.actions.insert(id, control.action.clone());
            ids.push(id);
        }
        if let Some((_, root)) = self.update.nodes.iter_mut().find(|(id, _)| *id == ROOT) {
            let mut children = root.children().to_vec();
            children.extend(ids);
            root.set_children(children);
        }
    }
}

const ROOT: NodeId = NodeId(1);
const CONTENT: NodeId = NodeId(2);
const GRID_CONTEXT: NodeId = NodeId(3);

fn stable_node(domain: u64, first: i64, second: i64) -> NodeId {
    // SplitMix-style integer mixing gives deterministic IDs without retaining
    // an unbounded registry. Domain separation prevents control/row aliases.
    let mut value = (first as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (second as u64).rotate_left(29)
        ^ domain;
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    NodeId(value.wrapping_mul(0x94d0_49bb_1331_11eb) | (1 << 63))
}

fn actionable(role: Role, label: String, disabled: bool, busy: bool) -> Node {
    let mut node = Node::new(role);
    node.set_label(label);
    if disabled {
        node.set_disabled();
    }
    if busy {
        node.set_busy();
    }
    if !disabled {
        node.add_action(Action::Click);
    }
    node
}

pub fn people_overview_snapshot(
    items: &[PersonAccessibilityItem<'_>],
) -> PeopleAccessibilitySnapshot {
    let mut actions = HashMap::new();
    let mut nodes = Vec::with_capacity(items.len() + 2);
    let children = items
        .iter()
        .map(|person| {
            let id = stable_node(0x10, person.id, 0);
            let label = format!(
                "{}, {} confirmed faces, {} suggestions",
                person.name, person.confirmed, person.suggested
            );
            nodes.push((id, actionable(Role::Button, label, false, false)));
            actions.insert(id, PeopleAccessibilityAction::OpenPerson(person.id));
            id
        })
        .collect::<Vec<_>>();
    let mut content = Node::new(Role::List);
    content.set_label("People");
    content.set_children(children);
    nodes.push((CONTENT, content));
    let mut root = Node::new(Role::Application);
    root.set_label("Rekesz People");
    root.set_children([CONTENT]);
    nodes.push((ROOT, root));
    PeopleAccessibilitySnapshot {
        update: TreeUpdate {
            nodes,
            tree: Some(TreeInfo::new(ROOT)),
            tree_id: TreeId::ROOT,
            focus: ROOT,
        },
        generation: 0,
        actions,
    }
}

pub fn people_faces_snapshot(
    items: &[FaceAccessibilityItem<'_>],
    default_context: bool,
) -> PeopleAccessibilitySnapshot {
    let mut actions = HashMap::new();
    let mut nodes = Vec::with_capacity(items.len() * 4 + 3);
    let mut children = vec![GRID_CONTEXT];
    let mut grid_toggle = actionable(
        Role::Switch,
        "Show full context for all face tiles".to_owned(),
        false,
        false,
    );
    grid_toggle.set_toggled(if default_context {
        Toggled::True
    } else {
        Toggled::False
    });
    nodes.push((GRID_CONTEXT, grid_toggle));
    actions.insert(GRID_CONTEXT, PeopleAccessibilityAction::ToggleGridContext);

    for face in items {
        let tile_id = stable_node(0x20, face.face_id, face.person_id);
        let toggle_id = stable_node(0x21, face.face_id, face.person_id);
        let accept_id = stable_node(0x22, face.face_id, face.person_id);
        let reject_id = stable_node(0x23, face.face_id, face.person_id);
        let mut tile_children = vec![toggle_id];
        if face.can_accept {
            tile_children.push(accept_id);
        }
        if face.can_reject {
            tile_children.push(reject_id);
        }
        let person = face.person_name.unwrap_or("unnamed person");
        let mut tile = actionable(
            Role::ListItem,
            format!("{}, {}, {}", face.file_name, person, face.decision),
            false,
            face.pending,
        );
        tile.set_selected(face.selected);
        tile.set_children(tile_children);
        nodes.push((tile_id, tile));
        actions.insert(
            tile_id,
            PeopleAccessibilityAction::SelectFace {
                face_id: face.face_id,
                person_id: face.person_id,
            },
        );
        let mut toggle = actionable(
            Role::Switch,
            format!("Show full context for {}", face.file_name),
            false,
            false,
        );
        toggle.set_toggled(if face.context {
            Toggled::True
        } else {
            Toggled::False
        });
        nodes.push((toggle_id, toggle));
        actions.insert(
            toggle_id,
            PeopleAccessibilityAction::ToggleFaceContext {
                face_id: face.face_id,
                person_id: face.person_id,
            },
        );
        if face.can_accept {
            nodes.push((
                accept_id,
                actionable(
                    Role::Button,
                    format!("Accept {} as {}", face.file_name, person),
                    face.pending,
                    face.pending,
                ),
            ));
            actions.insert(
                accept_id,
                PeopleAccessibilityAction::AcceptFace {
                    face_id: face.face_id,
                    person_id: face.person_id,
                },
            );
        }
        if face.can_reject {
            nodes.push((
                reject_id,
                actionable(
                    Role::Button,
                    format!("Reject {} for {}", face.file_name, person),
                    face.pending,
                    face.pending,
                ),
            ));
            actions.insert(
                reject_id,
                PeopleAccessibilityAction::RejectFace {
                    face_id: face.face_id,
                    person_id: face.person_id,
                },
            );
        }
        children.push(tile_id);
    }
    let mut content = Node::new(Role::List);
    content.set_label("Face review");
    content.set_children(children);
    nodes.push((CONTENT, content));
    let mut root = Node::new(Role::Application);
    root.set_label("Rekesz People face review");
    root.set_children([CONTENT]);
    nodes.push((ROOT, root));
    PeopleAccessibilitySnapshot {
        update: TreeUpdate {
            nodes,
            tree: Some(TreeInfo::new(ROOT)),
            tree_id: TreeId::ROOT,
            focus: ROOT,
        },
        generation: 0,
        actions,
    }
}

pub fn people_dialog_snapshot(label: &str) -> PeopleAccessibilitySnapshot {
    let mut root = Node::new(Role::Dialog);
    root.set_label(label);
    PeopleAccessibilitySnapshot {
        update: TreeUpdate {
            nodes: vec![(ROOT, root)],
            tree: Some(TreeInfo::new(ROOT)),
            tree_id: TreeId::ROOT,
            focus: ROOT,
        },
        generation: 0,
        actions: HashMap::new(),
    }
}

#[derive(Clone)]
struct QueuedRequest {
    generation: u64,
    request: ActionRequest,
}

// The accesskit_unix adapter is Linux-only; these handlers are only
// constructed from the adapter setup inside the Linux-gated blocks below.
#[cfg(target_os = "linux")]
struct Activation(Arc<Mutex<PeopleAccessibilitySnapshot>>);
#[cfg(target_os = "linux")]
impl accesskit::ActivationHandler for Activation {
    fn request_initial_tree(&mut self) -> Option<TreeUpdate> {
        Some(
            self.0
                .lock()
                .expect("accessibility snapshot poisoned")
                .update
                .clone(),
        )
    }
}

#[cfg(target_os = "linux")]
struct Actions {
    latest: Arc<Mutex<PeopleAccessibilitySnapshot>>,
    requests: Arc<Mutex<Vec<QueuedRequest>>>,
}
#[cfg(target_os = "linux")]
impl accesskit::ActionHandler for Actions {
    fn do_action(&mut self, request: ActionRequest) {
        let generation = self
            .latest
            .lock()
            .expect("accessibility snapshot poisoned")
            .generation;
        self.requests
            .lock()
            .expect("accessibility actions poisoned")
            .push(QueuedRequest {
                generation,
                request,
            });
    }
}

#[cfg(target_os = "linux")]
struct Deactivation;
#[cfg(target_os = "linux")]
impl accesskit::DeactivationHandler for Deactivation {
    fn deactivate_accessibility(&mut self) {}
}

pub struct PeopleAccessibilityBridge {
    latest: Arc<Mutex<PeopleAccessibilitySnapshot>>,
    requests: Arc<Mutex<Vec<QueuedRequest>>>,
    #[cfg(target_os = "linux")]
    adapter: Mutex<Option<accesskit_unix::Adapter>>,
}

impl PeopleAccessibilityBridge {
    pub fn new() -> Self {
        let latest = Arc::new(Mutex::new(people_overview_snapshot(&[])));
        let requests = Arc::new(Mutex::new(Vec::new()));
        Self {
            latest,
            requests,
            #[cfg(target_os = "linux")]
            adapter: Mutex::new(None),
        }
    }

    pub fn update(&self, snapshot: PeopleAccessibilitySnapshot, window_focused: bool) {
        *self.latest.lock().expect("accessibility snapshot poisoned") = snapshot.clone();
        // The focus state only feeds the Linux adapter; keep the parameter
        // "used" on other platforms.
        #[cfg(not(target_os = "linux"))]
        let _ = window_focused;
        #[cfg(target_os = "linux")]
        {
            let mut adapter = self.adapter.lock().expect("accessibility adapter poisoned");
            let adapter = adapter.get_or_insert_with(|| {
                accesskit_unix::Adapter::new(
                    Activation(Arc::clone(&self.latest)),
                    Actions {
                        latest: Arc::clone(&self.latest),
                        requests: Arc::clone(&self.requests),
                    },
                    Deactivation,
                )
            });
            adapter.update_window_focus_state(window_focused);
            adapter.update_if_active(|| snapshot.update);
        }
    }

    pub fn take_actions(&self) -> Vec<(u64, PeopleAccessibilityAction)> {
        let requests = std::mem::take(
            &mut *self
                .requests
                .lock()
                .expect("accessibility actions poisoned"),
        );
        let latest = self.latest.lock().expect("accessibility snapshot poisoned");
        requests
            .into_iter()
            .filter(|queued| queued.request.action == Action::Click)
            .filter_map(|queued| {
                latest
                    .action_for(queued.request.target_node)
                    .map(|action| (queued.generation, action))
            })
            .collect()
    }

    pub fn deactivate(&self) {
        self.requests
            .lock()
            .expect("accessibility actions poisoned")
            .clear();
        #[cfg(target_os = "linux")]
        {
            self.adapter
                .lock()
                .expect("accessibility adapter poisoned")
                .take();
        }
    }

    #[cfg(test)]
    pub fn request_click(&self, action: PeopleAccessibilityAction) {
        let latest = self.latest.lock().expect("accessibility snapshot poisoned");
        let node = latest
            .actions
            .iter()
            .find_map(|(node, candidate)| (*candidate == action).then_some(*node))
            .expect("requested accessibility action is absent from the current tree");
        let generation = latest.generation;
        drop(latest);
        self.requests
            .lock()
            .expect("accessibility actions poisoned")
            .push(QueuedRequest {
                generation,
                request: ActionRequest {
                    action: Action::Click,
                    target_tree: TreeId::ROOT,
                    target_node: node,
                    data: None,
                },
            });
    }

    #[cfg(test)]
    pub fn snapshot(&self) -> PeopleAccessibilitySnapshot {
        self.latest
            .lock()
            .expect("accessibility snapshot poisoned")
            .clone()
    }

    #[cfg(all(test, target_os = "linux"))]
    pub fn adapter_active(&self) -> bool {
        self.adapter
            .lock()
            .expect("accessibility adapter poisoned")
            .is_some()
    }
}

#[cfg(test)]
#[path = "accessibility/tests.rs"]
mod tests;
