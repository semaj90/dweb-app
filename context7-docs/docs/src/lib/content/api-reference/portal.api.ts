import { portalProps } from './shared';
import type { PortalProps } from "bits-ui";
import { defineUtilityApiSchema } from '../utils';

export const portal = defineUtilityApiSchema<PortalProps>({
	title: "Portal",
	description: "Renders the children to a different location in the DOM.",
	props: portalProps,
});
