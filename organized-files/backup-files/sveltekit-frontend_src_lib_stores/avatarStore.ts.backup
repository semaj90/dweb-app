import { writable } from 'svelte/store';
import { browser } from '$app/environment';

interface AvatarState {
	url: string | null;
	isUploading: boolean;
	error: string | null;
}

const initialState: AvatarState = {
	url: null,
	isUploading: false,
	error: null
};

function createAvatarStore() {
	const { subscribe, set, update } = writable<AvatarState>(initialState);

	return {
		subscribe,
		
		// Load avatar from local storage and API
		loadAvatar: async () => {
			if (!browser) return;
			
			// Try local storage first for instant loading
			const cachedAvatar = localStorage.getItem('user_avatar_url');
			if (cachedAvatar) {
				update(state => ({ ...state, url: cachedAvatar }));
			}

			// Then fetch from API for up-to-date data
			try {
				const response = await fetch('/api/user/profile');
				if (response.ok) {
					const data = await response.json();
					const avatarUrl = data.user?.avatarUrl || '/images/default-avatar.svg';
					
					// Update store and cache
					update(state => ({ ...state, url: avatarUrl, error: null }));
					localStorage.setItem('user_avatar_url', avatarUrl);
				}
			} catch (error) {
				console.error('Failed to load avatar:', error);
				update(state => ({ ...state, error: 'Failed to load avatar' }));
			}
		},

		// Upload new avatar
		uploadAvatar: async (file: File) => {
			update(state => ({ ...state, isUploading: true, error: null }));

			try {
				const formData = new FormData();
				formData.append('avatar', file);

				const response = await fetch('/api/user/avatar/upload', {
					method: 'POST',
					body: formData
				});

				const data = await response.json();

				if (response.ok) {
					const newAvatarUrl = data.avatarUrl;
					update(state => ({ 
						...state, 
						url: newAvatarUrl, 
						isUploading: false,
						error: null 
					}));
					
					// Update local storage
					if (browser) {
						localStorage.setItem('user_avatar_url', newAvatarUrl);
					}
					
					return { success: true, url: newAvatarUrl };
				} else {
					throw new Error(data.error || 'Upload failed');
				}
			} catch (error) {
				const errorMessage = error instanceof Error ? error.message : 'Upload failed';
				update(state => ({ 
					...state, 
					isUploading: false, 
					error: errorMessage 
				}));
				return { success: false, error: errorMessage };
			}
		},

		// Remove avatar
		removeAvatar: async () => {
			try {
				const response = await fetch('/api/user/avatar/upload', {
					method: 'DELETE'
				});

				if (response.ok) {
					const defaultAvatar = '/images/default-avatar.svg';
					update(state => ({ 
						...state, 
						url: defaultAvatar, 
						error: null 
					}));
					
					// Update local storage
					if (browser) {
						localStorage.setItem('user_avatar_url', defaultAvatar);
					}
					
					return { success: true };
				} else {
					throw new Error('Failed to remove avatar');
				}
			} catch (error) {
				const errorMessage = error instanceof Error ? error.message : 'Removal failed';
				update(state => ({ ...state, error: errorMessage }));
				return { success: false, error: errorMessage };
			}
		},

		// Clear error
		clearError: () => {
			update(state => ({ ...state, error: null }));
		},

		// Reset store
		reset: () => {
			set(initialState);
			if (browser) {
				localStorage.removeItem('user_avatar_url');
			}
		}
	};
}

export const avatarStore = createAvatarStore();
