"""Example: Generate an RCT study-design diagram for Adami et al. 2021 (VR training paper).

Pulled from Obsidian paper-library note: T1-XR-Training/
'Effectiveness of VR-based training on improving construction workers safety'
"""

import asyncio

from dotenv import load_dotenv

from paperbanana import DiagramType, GenerationInput, PaperBananaPipeline
from paperbanana.core.config import Settings

load_dotenv()


async def main():
    # Methodology distilled from the Obsidian note
    source_context = """
    We conducted a randomized controlled trial (RCT) to evaluate the effectiveness
    of Virtual Reality (VR)-based training compared to traditional in-person training
    for construction workers operating a demolition robot.

    Participants:
    - N = 50 construction workers
    - Randomly assigned to one of two training arms (1:1 allocation)

    Intervention Arms:
    - Arm A (VR Training): Head-mounted-display-based immersive training simulating
      the demolition-robot teleoperation task in a virtual construction site.
    - Arm B (In-Person Training): Traditional classroom + hands-on demonstration
      delivered by a certified instructor on the physical robot.

    Outcomes Measured (post-training):
    1. Knowledge acquisition — multiple-choice knowledge test.
    2. Operational skills — standardized task-performance assessment on the real robot.
    3. Safety behavior — observational checklist during task execution.

    Analysis: Between-group comparisons were performed using quantitative and
    qualitative analyses to contrast VR vs in-person outcomes across the three
    measures. VR-based training yielded significantly higher scores on all three
    outcomes.
    """

    caption = (
        "Study design of the randomized controlled trial comparing VR-based training "
        "versus in-person training for construction workers operating a demolition robot. "
        "N=50 workers were randomized 1:1, trained in their assigned modality, and "
        "evaluated on knowledge acquisition, operational skills, and safety behavior."
    )

    settings = Settings(
        vlm_provider="gemini",
        vlm_model="gemini-2.5-flash",
        image_provider="google_imagen",
        image_model="gemini-3-pro-image-preview",
        refinement_iterations=2,
    )

    pipeline = PaperBananaPipeline(settings=settings)

    result = await pipeline.generate(
        GenerationInput(
            source_context=source_context,
            communicative_intent=caption,
            diagram_type=DiagramType.METHODOLOGY,
        )
    )

    print(f"Generated diagram: {result.image_path}")
    print(f"Total iterations: {len(result.iterations)}")
    print(f"Run ID: {result.metadata.get('run_id')}")


if __name__ == "__main__":
    asyncio.run(main())
