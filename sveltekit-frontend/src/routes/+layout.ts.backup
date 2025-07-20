// Client-side layout load function to sync user data with store
import { setUser } from "../lib/stores/user";
import type { LayoutLoad } from './$types';

export const load: LayoutLoad = ({ data }) => {
    // 'data' comes from +layout.server.ts
    if (data.user) {
        setUser(data.user);
    } else {
        setUser(null);
    }

    return {
        user: data.user
    };
};
