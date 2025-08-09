// Slider Component Barrel Export
import { Slider } from "bits-ui";

export const SliderRoot = Slider.Root;
export const SliderRange = Slider.Range;
export const SliderThumb = Slider.Thumb;
export const SliderTick = Slider.Tick;

// Re-export the whole Slider module
export { Slider };

// TypeScript interface for Slider props
export interface SliderProps {
  value?: number[];
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
  onValueChange?: (value: number[]) => void;
  orientation?: "horizontal" | "vertical";
}
