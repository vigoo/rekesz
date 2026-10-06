use crate::{
    catalog::DevelopRecipe,
    formats::{is_heif, is_raw},
};

use super::{RenderIdentity, RenderMode, RenderRequest, RenderVariant, develop};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RenderStage {
    CacheLookup,
    Decode,
    RawPhotographic,
    PostDevelop,
    Resize,
    CacheWriteEligible,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DecodePlan {
    Standard { scaled_jpeg: bool },
    Heif,
    Raw,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ResizePlan {
    None,
    Derivative {
        max_width: u32,
        max_height: u32,
        before_develop: bool,
    },
}

/// Validated, ordered execution decisions compiled once per render request.
pub(super) struct RenderPlan {
    request: RenderRequest,
    decode: DecodePlan,
    resize: ResizePlan,
    recipe: Option<DevelopRecipe>,
    raw_recipe: Option<DevelopRecipe>,
    post_recipe: Option<DevelopRecipe>,
    raw_applied_actions: u64,
    raw_fidelity_warnings: u64,
    stages: Vec<RenderStage>,
    analysis_memory_budget: Option<usize>,
    lens_correction: bool,
}

impl RenderPlan {
    pub(super) fn compile(
        path: std::path::PathBuf,
        variant: RenderVariant,
        identity: RenderIdentity,
        recipe: Option<&DevelopRecipe>,
    ) -> Result<Self, String> {
        validate_variant(variant)?;
        let developed_recipe = if identity.mode == RenderMode::Developed {
            let recipe =
                recipe.ok_or_else(|| "developed render is missing its recipe".to_owned())?;
            recipe
                .validate()
                .map_err(|error| format!("invalid developed recipe: {error:#}"))?;
            if recipe.digest().ok() != Some(identity.recipe_digest) {
                return Err("developed render recipe digest mismatch".to_owned());
            }
            Some(recipe.clone())
        } else {
            None
        };
        let raw = is_raw(&path);
        let decode = if is_heif(&path) {
            DecodePlan::Heif
        } else if raw {
            DecodePlan::Raw
        } else {
            DecodePlan::Standard {
                scaled_jpeg: is_jpeg(&path)
                    && matches!(
                        variant,
                        RenderVariant::Thumbnail { .. } | RenderVariant::Preview { .. }
                    )
                    && identity.mode != RenderMode::Developed,
            }
        };
        let resize = match variant {
            RenderVariant::Full => ResizePlan::None,
            RenderVariant::Thumbnail { width } => ResizePlan::Derivative {
                max_width: width,
                max_height: width,
                before_develop: developed_recipe.is_none(),
            },
            RenderVariant::Preview {
                max_width,
                max_height,
            } => ResizePlan::Derivative {
                max_width,
                max_height,
                before_develop: developed_recipe.is_none(),
            },
            RenderVariant::FaceCrop { .. } => ResizePlan::None,
            RenderVariant::FaceContext { width, .. } => ResizePlan::Derivative {
                max_width: width,
                max_height: width,
                before_develop: developed_recipe.is_none(),
            },
        };
        let (raw_recipe, post_recipe, raw_applied_actions, raw_fidelity_warnings) =
            match (raw, developed_recipe.as_ref()) {
                (true, Some(recipe)) => {
                    let partition = partition_raw_recipe(recipe);
                    (
                        Some(recipe.clone()),
                        Some(partition.post_recipe),
                        partition.applied_actions,
                        partition.fidelity_warnings,
                    )
                }
                (false, Some(recipe)) => (None, Some(recipe.clone()), 0, 0),
                (_, None) => (None, None, 0, 0),
            };
        let mut stages = vec![RenderStage::CacheLookup, RenderStage::Decode];
        if raw {
            stages.push(RenderStage::RawPhotographic);
        }
        if developed_recipe.is_some() {
            stages.push(RenderStage::PostDevelop);
        }
        if variant != RenderVariant::Full {
            stages.push(RenderStage::Resize);
        }
        if variant != RenderVariant::Full {
            stages.push(RenderStage::CacheWriteEligible);
        }
        Ok(Self {
            request: RenderRequest::with_identity(path, variant, identity),
            decode,
            resize,
            recipe: developed_recipe,
            raw_recipe,
            post_recipe,
            raw_applied_actions,
            raw_fidelity_warnings,
            stages,
            analysis_memory_budget: None,
            lens_correction: false,
        })
    }

    pub(super) fn set_analysis_memory_budget(&mut self, budget: usize) {
        self.analysis_memory_budget = Some(budget);
    }

    /// Lens correction engages for photo renders; analysis and face crops
    /// must stay in the uncorrected (analysis-geometry) space.
    pub(super) fn set_lens_correction(&mut self, enabled: bool) {
        self.lens_correction = enabled;
    }

    pub(super) fn lens_correction(&self) -> bool {
        self.lens_correction
    }

    pub(super) fn analysis_memory_budget(&self) -> Option<usize> {
        self.analysis_memory_budget
    }

    pub(super) fn request(&self) -> &RenderRequest {
        &self.request
    }

    pub(super) fn decode(&self) -> DecodePlan {
        self.decode
    }

    pub(super) fn resize(&self) -> ResizePlan {
        self.resize
    }

    pub(super) fn is_raw_decode(&self) -> bool {
        self.decode == DecodePlan::Raw
    }

    pub(super) fn caches_decoded_source(&self) -> bool {
        self.decode == DecodePlan::Raw
            || matches!(
                self.request.variant,
                RenderVariant::FaceCrop { .. } | RenderVariant::FaceContext { .. }
            )
    }

    pub(super) fn raw_recipe(&self) -> Option<&DevelopRecipe> {
        self.raw_recipe.as_ref()
    }

    pub(super) fn recipe(&self) -> Option<&DevelopRecipe> {
        self.recipe.as_ref()
    }

    pub(super) fn post_recipe(&self) -> Option<&DevelopRecipe> {
        self.post_recipe.as_ref()
    }

    pub(super) fn raw_metadata(&self) -> (u64, u64) {
        (self.raw_applied_actions, self.raw_fidelity_warnings)
    }

    pub(super) fn includes(&self, stage: RenderStage) -> bool {
        self.stages.contains(&stage)
    }

    #[cfg(test)]
    pub(super) fn stages(&self) -> &[RenderStage] {
        &self.stages
    }
}

fn validate_variant(variant: RenderVariant) -> Result<(), String> {
    match variant {
        RenderVariant::Thumbnail { width }
        | RenderVariant::FaceContext { width }
        | RenderVariant::FaceCrop { width, .. }
            if width == 0 =>
        {
            Err("render width must be positive".to_owned())
        }
        RenderVariant::Preview {
            max_width,
            max_height,
        } if max_width == 0 || max_height == 0 => {
            Err("preview dimensions must be positive".to_owned())
        }
        RenderVariant::FaceCrop {
            left_millionths,
            top_millionths,
            right_millionths,
            bottom_millionths,
            padding_percent,
            ..
        } if left_millionths >= right_millionths
            || top_millionths >= bottom_millionths
            || right_millionths > 1_000_000
            || bottom_millionths > 1_000_000
            || padding_percent > 200 =>
        {
            Err("invalid face crop bounds or padding".to_owned())
        }
        _ => Ok(()),
    }
}

fn is_jpeg(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("jpg") || extension.eq_ignore_ascii_case("jpeg")
        })
}

struct RawRecipePartition {
    post_recipe: DevelopRecipe,
    applied_actions: u64,
    fidelity_warnings: u64,
}

/// Declares RAW-owned controls once: each group both contributes metadata and
/// is removed from the display-referred recipe passed to `develop`.
fn partition_raw_recipe(recipe: &DevelopRecipe) -> RawRecipePartition {
    let mut post_recipe = recipe.clone();
    let mut actions = 0;

    macro_rules! raw_group {
        ($action:expr; $($field:ident),+ $(,)?) => {
            if false $(|| recipe.$field.is_some())+ {
                actions |= $action;
            }
            $(post_recipe.$field = None;)+
        };
    }

    raw_group!(develop::actions::EXPOSURE; exposure_ev);
    raw_group!(develop::actions::WHITE_BALANCE; white_balance, white_balance_increment);
    raw_group!(develop::actions::BASIC_TONE; contrast, highlights, shadows, whites, blacks);
    raw_group!(develop::actions::COLOR; saturation, vibrance);
    raw_group!(develop::actions::TONE_CURVE;
        tone_curve, tone_curve_red, tone_curve_green, tone_curve_blue);
    raw_group!(develop::actions::DETAIL; detail);
    raw_group!(develop::actions::TEXTURE_DEHAZE; texture, dehaze);
    raw_group!(develop::actions::MONOCHROME; monochrome);
    raw_group!(develop::actions::PARAMETRIC_CURVE; parametric_curve);

    let approximated = actions
        & (develop::actions::WHITE_BALANCE
            | develop::actions::BASIC_TONE
            | develop::actions::COLOR
            | develop::actions::TONE_CURVE
            | develop::actions::DETAIL
            | develop::actions::TEXTURE_DEHAZE
            | develop::actions::MONOCHROME
            | develop::actions::PARAMETRIC_CURVE)
        != 0;
    RawRecipePartition {
        post_recipe,
        applied_actions: actions,
        fidelity_warnings: if approximated {
            develop::warnings::APPROXIMATED
        } else {
            0
        },
    }
}

#[cfg(test)]
mod tests;
