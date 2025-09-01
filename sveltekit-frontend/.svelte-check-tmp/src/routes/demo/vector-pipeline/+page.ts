import type { PageLoad } from './$types';

export const load: PageLoad = async () => {
  return {
    title: 'Vector Pipeline Demo',
    description: 'Interactive demonstration of the XState vector processing pipeline'
  };
};