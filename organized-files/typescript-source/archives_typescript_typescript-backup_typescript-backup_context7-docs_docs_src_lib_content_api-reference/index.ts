import { accordion } from './accordion.api';
import { alertDialog } from './alert-dialog.api';
import { aspectRatio } from './aspect-ratio.api';
import { avatar } from './avatar.api';
import { button } from './button.api';
import { calendar } from './calendar.api';
import { checkbox } from './checkbox.api';
import { collapsible } from './collapsible.api';
import { combobox } from './combobox.api';
import { command } from './command.api';
import { contextMenu } from './context-menu.api';
import { dateField } from './date-field.api';
import { datePicker } from './date-picker.api';
import { dateRangeField } from './date-range-field.api';
import { dateRangePicker } from './date-range-picker.api';
import { dialog } from './dialog.api';
import { dropdownMenu } from './dropdown-menu.api';
import { label } from './label.api';
import { linkPreview } from './link-preview.api';
import { navigationMenu } from './navigation-menu.api';
import { pagination } from './pagination.api';
import { pinInput } from './pin-input.api';
import { popover } from './popover.api';
import { progress } from './progress.api';
import { radioGroup } from './radio-group.api';
import { rangeCalendar } from './range-calendar.api';
import { ratingGroup } from './rating-group.api';
import { scrollArea } from './scroll-area.api';
import { select } from './select.api';
import { separator } from './separator.api';
import { slider } from './slider.api';
import { switchData } from './switch.api';
import { tabs } from './tabs.api';
import { toggleGroup } from './toggle-group.api';
import { toggle } from './toggle.api';
import { toolbar } from './toolbar.api';
import { tooltip } from './tooltip.api';
import { menubar } from './menubar.api';
import { meter } from './meter.api';
import { timeField } from './time-field.api';
import type { APISchema } from "$lib/types/index.js";
import { timeRangeField } from './time-range-field.api';
import { bitsConfig } from './bits-config.api';
import { portal } from './portal.api';

export const bits = [
	"accordion",
	"alert-dialog",
	"aspect-ratio",
	"avatar",
	"button",
	"calendar",
	"checkbox",
	"collapsible",
	"combobox",
	"command",
	"context-menu",
	"date-field",
	"date-picker",
	"date-range-field",
	"date-range-picker",
	"dialog",
	"dropdown-menu",
	"label",
	"link-preview",
	"menubar",
	"meter",
	"navigation-menu",
	"pagination",
	"pin-input",
	"popover",
	"progress",
	"radio-group",
	"range-calendar",
	"rating-group",
	"scroll-area",
	"select",
	"separator",
	"slider",
	"switch",
	"tabs",
	"time-field",
	"time-range-field",
	"toggle",
	"toggle-group",
	"toolbar",
	"tooltip",
] as const;

export const bitsSet = new Set(bits);

export function isBit(value: string): value is (typeof bits)[number] {
	return bitsSet.has(value as (typeof bits)[number]);
}

export type Bit = (typeof bits)[number];

export const apiSchemas: Record<Bit, APISchema[]> = {
	accordion,
	"alert-dialog": alertDialog,
	"aspect-ratio": aspectRatio,
	avatar,
	button,
	calendar,
	checkbox,
	collapsible,
	combobox,
	command,
	"context-menu": contextMenu,
	"date-field": dateField,
	"date-picker": datePicker,
	"date-range-field": dateRangeField,
	"date-range-picker": dateRangePicker,
	dialog,
	"dropdown-menu": dropdownMenu,
	label,
	"link-preview": linkPreview,
	menubar,
	meter,
	"navigation-menu": navigationMenu,
	pagination,
	"pin-input": pinInput,
	popover,
	progress,
	"radio-group": radioGroup,
	"range-calendar": rangeCalendar,
	"rating-group": ratingGroup,
	"scroll-area": scrollArea,
	select,
	separator,
	slider,
	switch: switchData,
	tabs,
	"time-field": timeField,
	"time-range-field": timeRangeField,
	toggle,
	"toggle-group": toggleGroup,
	toolbar,
	tooltip,
};

export const utilities = ["bits-config", "is-using-keyboard", "merge-props", "portal", "use-id"];

export const utilitiesSet = new Set(utilities);

export type Utility = (typeof utilities)[number];

export const utilitiesSchemas: Record<Utility, APISchema[]> = {
	"bits-config": [bitsConfig],
	"is-using-keyboard": [],
	"merge-props": [],
	portal: [portal],
	"use-id": [],
};

export function isUtility(value: string): value is Utility {
	return utilitiesSet.has(value as (typeof utilities)[number]);
}

export function getUtilityAPISchemas(utility: Utility): APISchema[] {
	return utilitiesSchemas[utility];
}

export function getAPISchemas(bit: Bit): APISchema[] {
	return apiSchemas[bit];
}

export * from './shared';
