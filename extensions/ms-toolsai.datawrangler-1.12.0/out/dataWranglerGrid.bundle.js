/*! For license information please see dataWranglerGrid.bundle.js.LICENSE.txt */
    <template class="${e=>e.circular?"circular":""}">
        <div class="control" part="control" style="${e=>e.generateBadgeStyle()}">
            <slot></slot>
        </div>
    </template>
`;class CSSDirective{createCSS(){return""}createBehavior(){}}function collectStyles(e,t){const n=[];let r="";const o=[];for(let i=0,a=e.length-1;i<a;++i){r+=e[i];let a=t[i];if(a instanceof CSSDirective){const e=a.createBehavior();a=a.createCSS(),e&&o.push(e)}a instanceof ElementStyles||a instanceof CSSStyleSheet?(""!==r.trim()&&(n.push(r),r=""),n.push(a)):r+=a}return r+=e[e.length-1],""!==r.trim()&&n.push(r),{styles:n,behaviors:o}}function css_css(e,...t){const{styles:n,behaviors:r}=collectStyles(e,t),o=ElementStyles.create(n);return r.length&&o.withBehaviors(...r),o}class CSSPartial extends CSSDirective{constructor(e,t){super(),this.behaviors=t,this.css="";const n=e.reduce(((e,t)=>("string"==typeof t?this.css+=t:e.push(t),e)),[]);n.length&&(this.styles=ElementStyles.create(n))}createBehavior(){return this}createCSS(){return this.css}bind(e){this.styles&&e.$fastController.addStyles(this.styles),this.behaviors.length&&e.$fastController.addBehaviors(this.behaviors)}unbind(e){this.styles&&e.$fastController.removeStyles(this.styles),this.behaviors.length&&e.$fastController.removeBehaviors(this.behaviors)}}function cssPartial(e,...t){const{styles:n,behaviors:r}=collectStyles(e,t);return new CSSPartial(n,r)}const display_hidden=":host([hidden]){display:none}";function display(e){return`${display_hidden}:host{display:${e}}`}function composedParent(e){const t=e.parentElement;if(t)return t;{const t=e.getRootNode();if(t.host instanceof HTMLElement)return t.host}return null}function composedContains(e,t){let n=t;for(;null!==n;){if(n===e)return!0;n=composedParent(n)}return!1}const defaultElement=document.createElement("div");function isFastElement(e){return e instanceof FASTElement}class QueuedStyleSheetTarget{setProperty(e,t){DOM.queueUpdate((()=>this.target.setProperty(e,t)))}removeProperty(e){DOM.queueUpdate((()=>this.target.removeProperty(e)))}}class ConstructableStyleSheetTarget extends QueuedStyleSheetTarget{constructor(e){super();const t=new CSSStyleSheet;this.target=t.cssRules[t.insertRule(":host{}")].style,e.$fastController.addStyles(ElementStyles.create([t]))}}class DocumentStyleSheetTarget extends QueuedStyleSheetTarget{constructor(){super();const e=new CSSStyleSheet;this.target=e.cssRules[e.insertRule(":root{}")].style,document.adoptedStyleSheets=[...document.adoptedStyleSheets,e]}}class HeadStyleElementStyleSheetTarget extends QueuedStyleSheetTarget{constructor(){super(),this.style=document.createElement("style"),document.head.appendChild(this.style);const{sheet:e}=this.style;if(e){const t=e.insertRule(":root{}",e.cssRules.length);this.target=e.cssRules[t].style}}}class StyleElementStyleSheetTarget{constructor(e){this.store=new Map,this.target=null;const t=e.$fastController;this.style=document.createElement("style"),t.addStyles(this.style),Observable.getNotifier(t).subscribe(this,"isConnected"),this.handleChange(t,"isConnected")}targetChanged(){if(null!==this.target)for(const[e,t]of this.store.entries())this.target.setProperty(e,t)}setProperty(e,t){this.store.set(e,t),DOM.queueUpdate((()=>{null!==this.target&&this.target.setProperty(e,t)}))}removeProperty(e){this.store.delete(e),DOM.queueUpdate((()=>{null!==this.target&&this.target.removeProperty(e)}))}handleChange(e,t){const{sheet:n}=this.style;if(n){const e=n.insertRule(":host{}",n.cssRules.length);this.target=n.cssRules[e].style}else this.target=null}}(0,tslib_es6.Cg)([observable],StyleElementStyleSheetTarget.prototype,"target",void 0);class ElementStyleSheetTarget{constructor(e){this.target=e.style}setProperty(e,t){DOM.queueUpdate((()=>this.target.setProperty(e,t)))}removeProperty(e){DOM.queueUpdate((()=>this.target.removeProperty(e)))}}class RootStyleSheetTarget{setProperty(e,t){RootStyleSheetTarget.properties[e]=t;for(const n of RootStyleSheetTarget.roots.values())PropertyTargetManager.getOrCreate(RootStyleSheetTarget.normalizeRoot(n)).setProperty(e,t)}removeProperty(e){delete RootStyleSheetTarget.properties[e];for(const t of RootStyleSheetTarget.roots.values())PropertyTargetManager.getOrCreate(RootStyleSheetTarget.normalizeRoot(t)).removeProperty(e)}static registerRoot(e){const{roots:t}=RootStyleSheetTarget;if(!t.has(e)){t.add(e);const n=PropertyTargetManager.getOrCreate(this.normalizeRoot(e));for(const e in RootStyleSheetTarget.properties)n.setProperty(e,RootStyleSheetTarget.properties[e])}}static unregisterRoot(e){const{roots:t}=RootStyleSheetTarget;if(t.has(e)){t.delete(e);const n=PropertyTargetManager.getOrCreate(RootStyleSheetTarget.normalizeRoot(e));for(const e in RootStyleSheetTarget.properties)n.removeProperty(e)}}static normalizeRoot(e){return e===defaultElement?document:e}}RootStyleSheetTarget.roots=new Set,RootStyleSheetTarget.properties={};const propertyTargetCache=new WeakMap,propertyTargetCtor=DOM.supportsAdoptedStyleSheets?ConstructableStyleSheetTarget:StyleElementStyleSheetTarget,PropertyTargetManager=Object.freeze({getOrCreate(e){if(propertyTargetCache.has(e))return propertyTargetCache.get(e);let t;return t=e===defaultElement?new RootStyleSheetTarget:e instanceof Document?DOM.supportsAdoptedStyleSheets?new DocumentStyleSheetTarget:new HeadStyleElementStyleSheetTarget:isFastElement(e)?new propertyTargetCtor(e):new ElementStyleSheetTarget(e),propertyTargetCache.set(e,t),t}});class DesignTokenImpl extends CSSDirective{constructor(e){super(),this.subscribers=new WeakMap,this._appliedTo=new Set,this.name=e.name,null!==e.cssCustomPropertyName&&(this.cssCustomProperty=`--${e.cssCustomPropertyName}`,this.cssVar=`var(${this.cssCustomProperty})`),this.id=DesignTokenImpl.uniqueId(),DesignTokenImpl.tokensById.set(this.id,this)}get appliedTo(){return[...this._appliedTo]}static from(e){return new DesignTokenImpl({name:"string"==typeof e?e:e.name,cssCustomPropertyName:"string"==typeof e?e:void 0===e.cssCustomPropertyName?e.name:e.cssCustomPropertyName})}static isCSSDesignToken(e){return"string"==typeof e.cssCustomProperty}static isDerivedDesignTokenValue(e){return"function"==typeof e}static getTokenById(e){return DesignTokenImpl.tokensById.get(e)}getOrCreateSubscriberSet(e=this){return this.subscribers.get(e)||this.subscribers.set(e,new Set)&&this.subscribers.get(e)}createCSS(){return this.cssVar||""}getValueFor(e){const t=DesignTokenNode.getOrCreate(e).get(this);if(void 0!==t)return t;throw new Error(`Value could not be retrieved for token named "${this.name}". Ensure the value is set for ${e} or an ancestor of ${e}.`)}setValueFor(e,t){return this._appliedTo.add(e),t instanceof DesignTokenImpl&&(t=this.alias(t)),DesignTokenNode.getOrCreate(e).set(this,t),this}deleteValueFor(e){return this._appliedTo.delete(e),DesignTokenNode.existsFor(e)&&DesignTokenNode.getOrCreate(e).delete(this),this}withDefault(e){return this.setValueFor(defaultElement,e),this}subscribe(e,t){const n=this.getOrCreateSubscriberSet(t);t&&!DesignTokenNode.existsFor(t)&&DesignTokenNode.getOrCreate(t),n.has(e)||n.add(e)}unsubscribe(e,t){const n=this.subscribers.get(t||this);n&&n.has(e)&&n.delete(e)}notify(e){const t=Object.freeze({token:this,target:e});this.subscribers.has(this)&&this.subscribers.get(this).forEach((e=>e.handleChange(t))),this.subscribers.has(e)&&this.subscribers.get(e).forEach((e=>e.handleChange(t)))}alias(e){return t=>e.getValueFor(t)}}DesignTokenImpl.uniqueId=(()=>{let e=0;return()=>(e++,e.toString(16))})(),DesignTokenImpl.tokensById=new Map;class CustomPropertyReflector{startReflection(e,t){e.subscribe(this,t),this.handleChange({token:e,target:t})}stopReflection(e,t){e.unsubscribe(this,t),this.remove(e,t)}handleChange(e){const{token:t,target:n}=e;this.add(t,n)}add(e,t){PropertyTargetManager.getOrCreate(t).setProperty(e.cssCustomProperty,this.resolveCSSValue(DesignTokenNode.getOrCreate(t).get(e)))}remove(e,t){PropertyTargetManager.getOrCreate(t).removeProperty(e.cssCustomProperty)}resolveCSSValue(e){return e&&"function"==typeof e.createCSS?e.createCSS():e}}class DesignTokenBindingObserver{constructor(e,t,n){this.source=e,this.token=t,this.node=n,this.dependencies=new Set,this.observer=Observable.binding(e,this,!1),this.observer.handleChange=this.observer.call,this.handleChange()}disconnect(){this.observer.disconnect()}handleChange(){this.node.store.set(this.token,this.observer.observe(this.node.target,defaultExecutionContext))}}class Store{constructor(){this.values=new Map}set(e,t){this.values.get(e)!==t&&(this.values.set(e,t),Observable.getNotifier(this).notify(e.id))}get(e){return Observable.track(this,e.id),this.values.get(e)}delete(e){this.values.delete(e)}all(){return this.values.entries()}}const nodeCache=new WeakMap,childToParent=new WeakMap;class DesignTokenNode{constructor(e){this.target=e,this.store=new Store,this.children=[],this.assignedValues=new Map,this.reflecting=new Set,this.bindingObservers=new Map,this.tokenValueChangeHandler={handleChange:(e,t)=>{const n=DesignTokenImpl.getTokenById(t);if(n&&(n.notify(this.target),DesignTokenImpl.isCSSDesignToken(n))){const t=this.parent,r=this.isReflecting(n);if(t){const o=t.get(n),i=e.get(n);o===i||r?o===i&&r&&this.stopReflectToCSS(n):this.reflectToCSS(n)}else r||this.reflectToCSS(n)}}},nodeCache.set(e,this),Observable.getNotifier(this.store).subscribe(this.tokenValueChangeHandler),e instanceof FASTElement?e.$fastController.addBehaviors([this]):e.isConnected&&this.bind()}static getOrCreate(e){return nodeCache.get(e)||new DesignTokenNode(e)}static existsFor(e){return nodeCache.has(e)}static findParent(e){if(defaultElement!==e.target){let t=composedParent(e.target);for(;null!==t;){if(nodeCache.has(t))return nodeCache.get(t);t=composedParent(t)}return DesignTokenNode.getOrCreate(defaultElement)}return null}static findClosestAssignedNode(e,t){let n=t;do{if(n.has(e))return n;n=n.parent?n.parent:n.target!==defaultElement?DesignTokenNode.getOrCreate(defaultElement):null}while(null!==n);return null}get parent(){return childToParent.get(this)||null}has(e){return this.assignedValues.has(e)}get(e){const t=this.store.get(e);if(void 0!==t)return t;const n=this.getRaw(e);return void 0!==n?(this.hydrate(e,n),this.get(e)):void 0}getRaw(e){var t;return this.assignedValues.has(e)?this.assignedValues.get(e):null===(t=DesignTokenNode.findClosestAssignedNode(e,this))||void 0===t?void 0:t.getRaw(e)}set(e,t){DesignTokenImpl.isDerivedDesignTokenValue(this.assignedValues.get(e))&&this.tearDownBindingObserver(e),this.assignedValues.set(e,t),DesignTokenImpl.isDerivedDesignTokenValue(t)?this.setupBindingObserver(e,t):this.store.set(e,t)}delete(e){this.assignedValues.delete(e),this.tearDownBindingObserver(e);const t=this.getRaw(e);t?this.hydrate(e,t):this.store.delete(e)}bind(){const e=DesignTokenNode.findParent(this);e&&e.appendChild(this);for(const e of this.assignedValues.keys())e.notify(this.target)}unbind(){this.parent&&childToParent.get(this).removeChild(this)}appendChild(e){e.parent&&childToParent.get(e).removeChild(e);const t=this.children.filter((t=>e.contains(t)));childToParent.set(e,this),this.children.push(e),t.forEach((t=>e.appendChild(t))),Observable.getNotifier(this.store).subscribe(e);for(const[t,n]of this.store.all())e.hydrate(t,this.bindingObservers.has(t)?this.getRaw(t):n)}removeChild(e){const t=this.children.indexOf(e);return-1!==t&&this.children.splice(t,1),Observable.getNotifier(this.store).unsubscribe(e),e.parent===this&&childToParent.delete(e)}contains(e){return composedContains(this.target,e.target)}reflectToCSS(e){this.isReflecting(e)||(this.reflecting.add(e),DesignTokenNode.cssCustomPropertyReflector.startReflection(e,this.target))}stopReflectToCSS(e){this.isReflecting(e)&&(this.reflecting.delete(e),DesignTokenNode.cssCustomPropertyReflector.stopReflection(e,this.target))}isReflecting(e){return this.reflecting.has(e)}handleChange(e,t){const n=DesignTokenImpl.getTokenById(t);n&&this.hydrate(n,this.getRaw(n))}hydrate(e,t){if(!this.has(e)){const n=this.bindingObservers.get(e);DesignTokenImpl.isDerivedDesignTokenValue(t)?n?n.source!==t&&(this.tearDownBindingObserver(e),this.setupBindingObserver(e,t)):this.setupBindingObserver(e,t):(n&&this.tearDownBindingObserver(e),this.store.set(e,t))}}setupBindingObserver(e,t){const n=new DesignTokenBindingObserver(t,e,this);return this.bindingObservers.set(e,n),n}tearDownBindingObserver(e){return!!this.bindingObservers.has(e)&&(this.bindingObservers.get(e).disconnect(),this.bindingObservers.delete(e),!0)}}function create(e){return DesignTokenImpl.from(e)}DesignTokenNode.cssCustomPropertyReflector=new CustomPropertyReflector,(0,tslib_es6.Cg)([observable],DesignTokenNode.prototype,"children",void 0);const DesignToken=Object.freeze({create,notifyConnection:e=>!(!e.isConnected||!DesignTokenNode.existsFor(e)||(DesignTokenNode.getOrCreate(e).bind(),0)),notifyDisconnection:e=>!(e.isConnected||!DesignTokenNode.existsFor(e)||(DesignTokenNode.getOrCreate(e).unbind(),0)),registerRoot(e=defaultElement){RootStyleSheetTarget.registerRoot(e)},unregisterRoot(e=defaultElement){RootStyleSheetTarget.unregisterRoot(e)}});function initThemeChangeListener(e){window.addEventListener("load",(()=>{new MutationObserver((()=>{applyCurrentTheme(e)})).observe(document.body,{attributes:!0,attributeFilter:["class"]}),applyCurrentTheme(e)}))}function applyCurrentTheme(e){const t=getComputedStyle(document.body),n=document.querySelector("body");if(n){const r=n.getAttribute("data-vscode-theme-kind");for(const[o,i]of e){let e=t.getPropertyValue(o).toString();"vscode-high-contrast"===r?(0===e.length&&i.name.includes("background")&&(e="transparent"),"button-icon-hover-background"===i.name&&(e="transparent")):"contrast-active-border"===i.name&&(e="transparent"),i.setValueFor(n,e)}}}const tokenMappings=new Map;let isThemeListenerInitialized=!1;function create_create(e,t){const n=DesignToken.create(e);return t&&(t.includes("--fake-vscode-token")&&(t=`${t}-${"id"+Math.random().toString(16).slice(2)}`),tokenMappings.set(t,n)),isThemeListenerInitialized||(initThemeChangeListener(tokenMappings),isThemeListenerInitialized=!0),n}const background=create_create("background","--vscode-editor-background").withDefault("#1e1e1e"),borderWidth=create_create("border-width").withDefault(1),contrastActiveBorder=create_create("contrast-active-border","--vscode-contrastActiveBorder").withDefault("#f38518"),contrastBorder=create_create("contrast-border","--vscode-contrastBorder").withDefault("#6fc3df"),cornerRadius=create_create("corner-radius").withDefault(0),designUnit=create_create("design-unit").withDefault(4),disabledOpacity=create_create("disabled-opacity").withDefault(.4),focusBorder=create_create("focus-border","--vscode-focusBorder").withDefault("#007fd4"),fontFamily=create_create("font-family","--vscode-font-family").withDefault("-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif, Apple Color Emoji, Segoe UI Emoji, Segoe UI Symbol"),fontWeight=create_create("font-weight","--vscode-font-weight").withDefault("400"),foreground=create_create("foreground","--vscode-foreground").withDefault("#cccccc"),inputHeight=create_create("input-height").withDefault("26"),inputMinWidth=create_create("input-min-width").withDefault("100px"),typeRampBaseFontSize=create_create("type-ramp-base-font-size","--vscode-font-size").withDefault("13px"),typeRampBaseLineHeight=create_create("type-ramp-base-line-height").withDefault("normal"),typeRampMinus1FontSize=create_create("type-ramp-minus1-font-size").withDefault("11px"),typeRampMinus1LineHeight=create_create("type-ramp-minus1-line-height").withDefault("16px"),typeRampMinus2FontSize=create_create("type-ramp-minus2-font-size").withDefault("9px"),typeRampMinus2LineHeight=create_create("type-ramp-minus2-line-height").withDefault("16px"),typeRampPlus1FontSize=create_create("type-ramp-plus1-font-size").withDefault("16px"),typeRampPlus1LineHeight=create_create("type-ramp-plus1-line-height").withDefault("24px"),scrollbarWidth=create_create("scrollbarWidth").withDefault("10px"),scrollbarHeight=create_create("scrollbarHeight").withDefault("10px"),scrollbarSliderBackground=create_create("scrollbar-slider-background","--vscode-scrollbarSlider-background").withDefault("#79797966"),scrollbarSliderHoverBackground=create_create("scrollbar-slider-hover-background","--vscode-scrollbarSlider-hoverBackground").withDefault("#646464b3"),scrollbarSliderActiveBackground=create_create("scrollbar-slider-active-background","--vscode-scrollbarSlider-activeBackground").withDefault("#bfbfbf66"),badgeBackground=create_create("badge-background","--vscode-badge-background").withDefault("#4d4d4d"),badgeForeground=create_create("badge-foreground","--vscode-badge-foreground").withDefault("#ffffff"),buttonBorder=create_create("button-border","--vscode-button-border").withDefault("transparent"),buttonIconBackground=create_create("button-icon-background").withDefault("transparent"),buttonIconCornerRadius=create_create("button-icon-corner-radius").withDefault("5px"),buttonIconFocusBorderOffset=create_create("button-icon-outline-offset").withDefault(0),buttonIconHoverBackground=create_create("button-icon-hover-background","--fake-vscode-token").withDefault("rgba(90, 93, 94, 0.31)"),buttonIconPadding=create_create("button-icon-padding").withDefault("3px"),buttonPrimaryBackground=create_create("button-primary-background","--vscode-button-background").withDefault("#0e639c"),buttonPrimaryForeground=create_create("button-primary-foreground","--vscode-button-foreground").withDefault("#ffffff"),buttonPrimaryHoverBackground=create_create("button-primary-hover-background","--vscode-button-hoverBackground").withDefault("#1177bb"),buttonSecondaryBackground=create_create("button-secondary-background","--vscode-button-secondaryBackground").withDefault("#3a3d41"),buttonSecondaryForeground=create_create("button-secondary-foreground","--vscode-button-secondaryForeground").withDefault("#ffffff"),buttonSecondaryHoverBackground=create_create("button-secondary-hover-background","--vscode-button-secondaryHoverBackground").withDefault("#45494e"),buttonPaddingHorizontal=create_create("button-padding-horizontal").withDefault("11px"),buttonPaddingVertical=create_create("button-padding-vertical").withDefault("4px"),checkboxBackground=create_create("checkbox-background","--vscode-checkbox-background").withDefault("#3c3c3c"),checkboxBorder=create_create("checkbox-border","--vscode-checkbox-border").withDefault("#3c3c3c"),checkboxCornerRadius=create_create("checkbox-corner-radius").withDefault(3),checkboxForeground=create_create("checkbox-foreground","--vscode-checkbox-foreground").withDefault("#f0f0f0"),listActiveSelectionBackground=create_create("list-active-selection-background","--vscode-list-activeSelectionBackground").withDefault("#094771"),listActiveSelectionForeground=create_create("list-active-selection-foreground","--vscode-list-activeSelectionForeground").withDefault("#ffffff"),listHoverBackground=create_create("list-hover-background","--vscode-list-hoverBackground").withDefault("#2a2d2e"),dividerBackground=create_create("divider-background","--vscode-settings-dropdownListBorder").withDefault("#454545"),dropdownBackground=create_create("dropdown-background","--vscode-dropdown-background").withDefault("#3c3c3c"),dropdownBorder=create_create("dropdown-border","--vscode-dropdown-border").withDefault("#3c3c3c"),dropdownForeground=create_create("dropdown-foreground","--vscode-dropdown-foreground").withDefault("#f0f0f0"),dropdownListMaxHeight=create_create("dropdown-list-max-height").withDefault("200px"),inputBackground=create_create("input-background","--vscode-input-background").withDefault("#3c3c3c"),inputForeground=create_create("input-foreground","--vscode-input-foreground").withDefault("#cccccc"),inputPlaceholderForeground=create_create("input-placeholder-foreground","--vscode-input-placeholderForeground").withDefault("#cccccc"),linkActiveForeground=create_create("link-active-foreground","--vscode-textLink-activeForeground").withDefault("#3794ff"),linkForeground=create_create("link-foreground","--vscode-textLink-foreground").withDefault("#3794ff"),progressBackground=create_create("progress-background","--vscode-progressBar-background").withDefault("#0e70c0"),panelTabActiveBorder=create_create("panel-tab-active-border","--vscode-panelTitle-activeBorder").withDefault("#e7e7e7"),panelTabActiveForeground=create_create("panel-tab-active-foreground","--vscode-panelTitle-activeForeground").withDefault("#e7e7e7"),panelTabForeground=create_create("panel-tab-foreground","--vscode-panelTitle-inactiveForeground").withDefault("#e7e7e799"),panelViewBackground=create_create("panel-view-background","--vscode-panel-background").withDefault("#1e1e1e"),panelViewBorder=create_create("panel-view-border","--vscode-panel-border").withDefault("#80808059"),tagCornerRadius=create_create("tag-corner-radius").withDefault("2px"),badgeStyles=(e,t)=>css_css`
	${display("inline-block")} :host {
		box-sizing: border-box;
		font-family: ${fontFamily};
		font-size: ${typeRampMinus1FontSize};
		line-height: ${typeRampMinus1LineHeight};
		text-align: center;
	}
	.control {
		align-items: center;
		background-color: ${badgeBackground};
		border: calc(${borderWidth} * 1px) solid ${buttonBorder};
		border-radius: 11px;
		box-sizing: border-box;
		color: ${badgeForeground};
		display: flex;
		height: calc(${designUnit} * 4px);
		justify-content: center;
		min-width: calc(${designUnit} * 4px + 2px);
		min-height: calc(${designUnit} * 4px + 2px);
		padding: 3px 6px;
	}
`;class Badge extends badge_Badge{connectedCallback(){super.connectedCallback(),this.circular||(this.circular=!0)}}const vsCodeBadge=Badge.compose({baseName:"badge",template:badgeTemplate,styles:badgeStyles});class ARIAGlobalStatesAndProperties{}(0,tslib_es6.Cg)([attr({attribute:"aria-atomic"})],ARIAGlobalStatesAndProperties.prototype,"ariaAtomic",void 0),(0,tslib_es6.Cg)([attr({attribute:"aria-busy"})],ARIAGlobalStatesAndProperties.prototype,"ariaBusy",void 0),(0,tslib_es6.Cg)([attr({attribute:"aria-controls"})],ARIAGlobalStatesAndProperties.prototype,"ariaControls",void 0),(0,tslib_es6.Cg)([attr({attribute:"aria-current"})],ARIAGlobalStatesAndProperties.prototype,"ariaCurrent",void 0),(0,tslib_es6.Cg)([attr({attribute:"aria-describedby"})],ARIAGlobalStatesAndProperties.prototype,"ariaDescribedby",void 0),(0,tslib_es6.Cg)([attr({attribute:"aria-details"})],ARIAGlobalStatesAndProperties.prototype,"ariaDetails",void 0),(0,tslib_es6.Cg)([attr({attribute:"aria-disabled"})],ARIAGlobalStatesAndProperties.prototype,"ariaDisabled",void 0),(0,tslib_es6.Cg)([attr({attribute:"aria-errormessage"})],ARIAGlobalStatesAndProperties.prototype,"ariaErrormessage",void 0),(0,tslib_es6.Cg)([attr({attribute:"aria-flowto"})],ARIAGlobalStatesAndProperties.prototype,"ariaFlowto",void 0),(0,tslib_es6.Cg)([attr({attribute:"aria-haspopup"})],ARIAGlobalStatesAndProperties.prototype,"ariaHaspopup",void 0),(0,tslib_es6.Cg)([attr({attribute:"aria-hidden"})],ARIAGlobalStatesAndProperties.prototype,"ariaHidden",void 0),(0,tslib_es6.Cg)([attr({attribute:"aria-invalid"})],ARIAGlobalStatesAndProperties.prototype,"ariaInvalid",void 0),(0,tslib_es6.Cg)([attr({attribute:"aria-keyshortcuts"})],ARIAGlobalStatesAndProperties.prototype,"ariaKeyshortcuts",void 0),(0,tslib_es6.Cg)([attr({attribute:"aria-label"})],ARIAGlobalStatesAndProperties.prototype,"ariaLabel",void 0),(0,tslib_es6.Cg)([attr({attribute:"aria-labelledby"})],ARIAGlobalStatesAndProperties.prototype,"ariaLabelledby",void 0),(0,tslib_es6.Cg)([attr({attribute:"aria-live"})],ARIAGlobalStatesAndProperties.prototype,"ariaLive",void 0),(0,tslib_es6.Cg)([attr({attribute:"aria-owns"})],ARIAGlobalStatesAndProperties.prototype,"ariaOwns",void 0),(0,tslib_es6.Cg)([attr({attribute:"aria-relevant"})],ARIAGlobalStatesAndProperties.prototype,"ariaRelevant",void 0),(0,tslib_es6.Cg)([attr({attribute:"aria-roledescription"})],ARIAGlobalStatesAndProperties.prototype,"ariaRoledescription",void 0);class RefBehavior{constructor(e,t){this.target=e,this.propertyName=t}bind(e){e[this.propertyName]=this.target}unbind(){}}function ref(e){return new AttachedBehaviorHTMLDirective("fast-ref",RefBehavior,e)}class StartEnd{handleStartContentChange(){this.startContainer.classList.toggle("start",this.start.assignedNodes().length>0)}handleEndContentChange(){this.endContainer.classList.toggle("end",this.end.assignedNodes().length>0)}}const endSlotTemplate=(e,t)=>html`
    <span
        part="end"
        ${ref("endContainer")}
        class=${e=>t.end?"end":void 0}
    >
        <slot name="end" ${ref("end")} @slotchange="${e=>e.handleEndContentChange()}">
            ${t.end||""}
        </slot>
    </span>
`,startSlotTemplate=(e,t)=>html`
    <span
        part="start"
        ${ref("startContainer")}
        class="${e=>t.start?"start":void 0}"
    >
        <slot
            name="start"
            ${ref("start")}
            @slotchange="${e=>e.handleStartContentChange()}"
        >
            ${t.start||""}
        </slot>
    </span>
`,endTemplate=html`
    <span part="end" ${ref("endContainer")}>
        <slot
            name="end"
            ${ref("end")}
            @slotchange="${e=>e.handleEndContentChange()}"
        ></slot>
    </span>
`,startTemplate=html`
    <span part="start" ${ref("startContainer")}>
        <slot
            name="start"
            ${ref("start")}
            @slotchange="${e=>e.handleStartContentChange()}"
        ></slot>
    </span>
`;function applyMixins(e,...t){t.forEach((t=>{if(Object.getOwnPropertyNames(t.prototype).forEach((n=>{"constructor"!==n&&Object.defineProperty(e.prototype,n,Object.getOwnPropertyDescriptor(t.prototype,n))})),t.attributes){const n=e.attributes||[];e.attributes=n.concat(t.attributes)}}))}var key_codes_KeyCodes;!function(e){e[e.alt=18]="alt",e[e.arrowDown=40]="arrowDown",e[e.arrowLeft=37]="arrowLeft",e[e.arrowRight=39]="arrowRight",e[e.arrowUp=38]="arrowUp",e[e.back=8]="back",e[e.backSlash=220]="backSlash",e[e.break=19]="break",e[e.capsLock=20]="capsLock",e[e.closeBracket=221]="closeBracket",e[e.colon=186]="colon",e[e.colon2=59]="colon2",e[e.comma=188]="comma",e[e.ctrl=17]="ctrl",e[e.delete=46]="delete",e[e.end=35]="end",e[e.enter=13]="enter",e[e.equals=187]="equals",e[e.equals2=61]="equals2",e[e.equals3=107]="equals3",e[e.escape=27]="escape",e[e.forwardSlash=191]="forwardSlash",e[e.function1=112]="function1",e[e.function10=121]="function10",e[e.function11=122]="function11",e[e.function12=123]="function12",e[e.function2=113]="function2",e[e.function3=114]="function3",e[e.function4=115]="function4",e[e.function5=116]="function5",e[e.function6=117]="function6",e[e.function7=118]="function7",e[e.function8=119]="function8",e[e.function9=120]="function9",e[e.home=36]="home",e[e.insert=45]="insert",e[e.menu=93]="menu",e[e.minus=189]="minus",e[e.minus2=109]="minus2",e[e.numLock=144]="numLock",e[e.numPad0=96]="numPad0",e[e.numPad1=97]="numPad1",e[e.numPad2=98]="numPad2",e[e.numPad3=99]="numPad3",e[e.numPad4=100]="numPad4",e[e.numPad5=101]="numPad5",e[e.numPad6=102]="numPad6",e[e.numPad7=103]="numPad7",e[e.numPad8=104]="numPad8",e[e.numPad9=105]="numPad9",e[e.numPadDivide=111]="numPadDivide",e[e.numPadDot=110]="numPadDot",e[e.numPadMinus=109]="numPadMinus",e[e.numPadMultiply=106]="numPadMultiply",e[e.numPadPlus=107]="numPadPlus",e[e.openBracket=219]="openBracket",e[e.pageDown=34]="pageDown",e[e.pageUp=33]="pageUp",e[e.period=190]="period",e[e.print=44]="print",e[e.quote=222]="quote",e[e.scrollLock=145]="scrollLock",e[e.shift=16]="shift",e[e.space=32]="space",e[e.tab=9]="tab",e[e.tilde=192]="tilde",e[e.windowsLeft=91]="windowsLeft",e[e.windowsOpera=219]="windowsOpera",e[e.windowsRight=92]="windowsRight"}(key_codes_KeyCodes||(key_codes_KeyCodes={}));const keyCodeAlt=18,keyCodeArrowDown=40,keyCodeArrowLeft=37,keyCodeArrowRight=39,keyCodeArrowUp=38,keyCodeBack=8,keyCodeBackSlash=220,keyCodeBreak=19,keyCodeCapsLock=20,keyCodeCloseBracket=221,keyCodeColon=186,keyCodeColon2=59,keyCodeComma=188,keyCodeCtrl=17,keyCodeDelete=46,keyCodeEnd=35,keyCodeEnter=13,keyCodeEquals=187,keyCodeEquals2=61,keyCodeEquals3=107,keyCodeEscape=27,keyCodeForwardSlash=191,keyCodeFunction1=112,keyCodeFunction10=121,keyCodeFunction11=122,keyCodeFunction12=123,keyCodeFunction2=113,keyCodeFunction3=114,keyCodeFunction4=115,keyCodeFunction5=116,keyCodeFunction6=117,keyCodeFunction7=118,keyCodeFunction8=119,keyCodeFunction9=120,keyCodeHome=36,keyCodeInsert=45,keyCodeMenu=93,keyCodeMinus=189,keyCodeMinus2=109,keyCodeNumLock=144,keyCodeNumPad0=96,keyCodeNumPad1=97,keyCodeNumPad2=98,keyCodeNumPad3=99,keyCodeNumPad4=100,keyCodeNumPad5=101,keyCodeNumPad6=102,keyCodeNumPad7=103,keyCodeNumPad8=104,keyCodeNumPad9=105,keyCodeNumPadDivide=111,keyCodeNumPadDot=110,keyCodeNumPadMinus=109,keyCodeNumPadMultiply=106,keyCodeNumPadPlus=107,keyCodeOpenBracket=219,keyCodePageDown=34,keyCodePageUp=33,keyCodePeriod=190,keyCodePrint=44,keyCodeQuote=222,keyCodeScrollLock=145,keyCodeShift=16,keyCodeSpace=32,keyCodeTab=9,keyCodeTilde=192,keyCodeWindowsLeft=91,keyCodeWindowsOpera=219,keyCodeWindowsRight=92,keyArrowDown="ArrowDown",keyArrowLeft="ArrowLeft",keyArrowRight="ArrowRight",keyArrowUp="ArrowUp",keyEnter="Enter",keyEscape="Escape",keyHome="Home",keyEnd="End",keyFunction2="F2",keyPageDown="PageDown",keyPageUp="PageUp",keySpace=" ",keyTab="Tab",keyBackspace="Backspace",keyDelete="Delete",ArrowKeys={ArrowDown:keyArrowDown,ArrowLeft:keyArrowLeft,ArrowRight:keyArrowRight,ArrowUp:keyArrowUp},proxySlotName="form-associated-proxy",ElementInternalsKey="ElementInternals",supportsElementInternals=ElementInternalsKey in window&&"setFormValue"in window[ElementInternalsKey].prototype,InternalsMap=new WeakMap;function FormAssociated(e){const t=class extends e{constructor(...e){super(...e),this.dirtyValue=!1,this.disabled=!1,this.proxyEventsToBlock=["change","click"],this.proxyInitialized=!1,this.required=!1,this.initialValue=this.initialValue||"",this.elementInternals||(this.formResetCallback=this.formResetCallback.bind(this))}static get formAssociated(){return supportsElementInternals}get validity(){return this.elementInternals?this.elementInternals.validity:this.proxy.validity}get form(){return this.elementInternals?this.elementInternals.form:this.proxy.form}get validationMessage(){return this.elementInternals?this.elementInternals.validationMessage:this.proxy.validationMessage}get willValidate(){return this.elementInternals?this.elementInternals.willValidate:this.proxy.willValidate}get labels(){if(this.elementInternals)return Object.freeze(Array.from(this.elementInternals.labels));if(this.proxy instanceof HTMLElement&&this.proxy.ownerDocument&&this.id){const e=this.proxy.labels,t=Array.from(this.proxy.getRootNode().querySelectorAll(`[for='${this.id}']`)),n=e?t.concat(Array.from(e)):t;return Object.freeze(n)}return emptyArray}valueChanged(e,t){this.dirtyValue=!0,this.proxy instanceof HTMLElement&&(this.proxy.value=this.value),this.currentValue=this.value,this.setFormValue(this.value),this.validate()}currentValueChanged(){this.value=this.currentValue}initialValueChanged(e,t){this.dirtyValue||(this.value=this.initialValue,this.dirtyValue=!1)}disabledChanged(e,t){this.proxy instanceof HTMLElement&&(this.proxy.disabled=this.disabled),DOM.queueUpdate((()=>this.classList.toggle("disabled",this.disabled)))}nameChanged(e,t){this.proxy instanceof HTMLElement&&(this.proxy.name=this.name)}requiredChanged(e,t){this.proxy instanceof HTMLElement&&(this.proxy.required=this.required),DOM.queueUpdate((()=>this.classList.toggle("required",this.required))),this.validate()}get elementInternals(){if(!supportsElementInternals)return null;let e=InternalsMap.get(this);return e||(e=this.attachInternals(),InternalsMap.set(this,e)),e}connectedCallback(){super.connectedCallback(),this.addEventListener("keypress",this._keypressHandler),this.value||(this.value=this.initialValue,this.dirtyValue=!1),this.elementInternals||(this.attachProxy(),this.form&&this.form.addEventListener("reset",this.formResetCallback))}disconnectedCallback(){this.proxyEventsToBlock.forEach((e=>this.proxy.removeEventListener(e,this.stopPropagation))),!this.elementInternals&&this.form&&this.form.removeEventListener("reset",this.formResetCallback)}checkValidity(){return this.elementInternals?this.elementInternals.checkValidity():this.proxy.checkValidity()}reportValidity(){return this.elementInternals?this.elementInternals.reportValidity():this.proxy.reportValidity()}setValidity(e,t,n){this.elementInternals?this.elementInternals.setValidity(e,t,n):"string"==typeof t&&this.proxy.setCustomValidity(t)}formDisabledCallback(e){this.disabled=e}formResetCallback(){this.value=this.initialValue,this.dirtyValue=!1}attachProxy(){var e;this.proxyInitialized||(this.proxyInitialized=!0,this.proxy.style.display="none",this.proxyEventsToBlock.forEach((e=>this.proxy.addEventListener(e,this.stopPropagation))),this.proxy.disabled=this.disabled,this.proxy.required=this.required,"string"==typeof this.name&&(this.proxy.name=this.name),"string"==typeof this.value&&(this.proxy.value=this.value),this.proxy.setAttribute("slot",proxySlotName),this.proxySlot=document.createElement("slot"),this.proxySlot.setAttribute("name",proxySlotName)),null===(e=this.shadowRoot)||void 0===e||e.appendChild(this.proxySlot),this.appendChild(this.proxy)}detachProxy(){var e;this.removeChild(this.proxy),null===(e=this.shadowRoot)||void 0===e||e.removeChild(this.proxySlot)}validate(e){this.proxy instanceof HTMLElement&&this.setValidity(this.proxy.validity,this.proxy.validationMessage,e)}setFormValue(e,t){this.elementInternals&&this.elementInternals.setFormValue(e,t||e)}_keypressHandler(e){if(e.key===keyEnter&&this.form instanceof HTMLFormElement){const e=this.form.querySelector("[type=submit]");null==e||e.click()}}stopPropagation(e){e.stopPropagation()}};return attr({mode:"boolean"})(t.prototype,"disabled"),attr({mode:"fromView",attribute:"value"})(t.prototype,"initialValue"),attr({attribute:"current-value"})(t.prototype,"currentValue"),attr(t.prototype,"name"),attr({mode:"boolean"})(t.prototype,"required"),observable(t.prototype,"value"),t}function CheckableFormAssociated(e){class t extends(FormAssociated(e)){}class n extends t{constructor(...e){super(e),this.dirtyChecked=!1,this.checkedAttribute=!1,this.checked=!1,this.dirtyChecked=!1}checkedAttributeChanged(){this.defaultChecked=this.checkedAttribute}defaultCheckedChanged(){this.dirtyChecked||(this.checked=this.defaultChecked,this.dirtyChecked=!1)}checkedChanged(e,t){this.dirtyChecked||(this.dirtyChecked=!0),this.currentChecked=this.checked,this.updateForm(),this.proxy instanceof HTMLInputElement&&(this.proxy.checked=this.checked),void 0!==e&&this.$emit("change"),this.validate()}currentCheckedChanged(e,t){this.checked=this.currentChecked}updateForm(){const e=this.checked?this.value:null;this.setFormValue(e,e)}connectedCallback(){super.connectedCallback(),this.updateForm()}formResetCallback(){super.formResetCallback(),this.checked=!!this.checkedAttribute,this.dirtyChecked=!1}}return attr({attribute:"checked",mode:"boolean"})(n.prototype,"checkedAttribute"),attr({attribute:"current-checked",converter:booleanConverter})(n.prototype,"currentChecked"),observable(n.prototype,"defaultChecked"),observable(n.prototype,"checked"),n}class _Button extends FoundationElement{}class FormAssociatedButton extends(FormAssociated(_Button)){constructor(){super(...arguments),this.proxy=document.createElement("input")}}class button_Button extends FormAssociatedButton{constructor(){super(...arguments),this.handleClick=e=>{var t;this.disabled&&(null===(t=this.defaultSlottedContent)||void 0===t?void 0:t.length)<=1&&e.stopPropagation()},this.handleSubmission=()=>{if(!this.form)return;const e=this.proxy.isConnected;e||this.attachProxy(),"function"==typeof this.form.requestSubmit?this.form.requestSubmit(this.proxy):this.proxy.click(),e||this.detachProxy()},this.handleFormReset=()=>{var e;null===(e=this.form)||void 0===e||e.reset()},this.handleUnsupportedDelegatesFocus=()=>{var e;window.ShadowRoot&&!window.ShadowRoot.prototype.hasOwnProperty("delegatesFocus")&&(null===(e=this.$fastController.definition.shadowOptions)||void 0===e?void 0:e.delegatesFocus)&&(this.focus=()=>{this.control.focus()})}}formactionChanged(){this.proxy instanceof HTMLInputElement&&(this.proxy.formAction=this.formaction)}formenctypeChanged(){this.proxy instanceof HTMLInputElement&&(this.proxy.formEnctype=this.formenctype)}formmethodChanged(){this.proxy instanceof HTMLInputElement&&(this.proxy.formMethod=this.formmethod)}formnovalidateChanged(){this.proxy instanceof HTMLInputElement&&(this.proxy.formNoValidate=this.formnovalidate)}formtargetChanged(){this.proxy instanceof HTMLInputElement&&(this.proxy.formTarget=this.formtarget)}typeChanged(e,t){this.proxy instanceof HTMLInputElement&&(this.proxy.type=this.type),"submit"===t&&this.addEventListener("click",this.handleSubmission),"submit"===e&&this.removeEventListener("click",this.handleSubmission),"reset"===t&&this.addEventListener("click",this.handleFormReset),"reset"===e&&this.removeEventListener("click",this.handleFormReset)}validate(){super.validate(this.control)}connectedCallback(){var e;super.connectedCallback(),this.proxy.setAttribute("type",this.type),this.handleUnsupportedDelegatesFocus();const t=Array.from(null===(e=this.control)||void 0===e?void 0:e.children);t&&t.forEach((e=>{e.addEventListener("click",this.handleClick)}))}disconnectedCallback(){var e;super.disconnectedCallback();const t=Array.from(null===(e=this.control)||void 0===e?void 0:e.children);t&&t.forEach((e=>{e.removeEventListener("click",this.handleClick)}))}}(0,tslib_es6.Cg)([attr({mode:"boolean"})],button_Button.prototype,"autofocus",void 0),(0,tslib_es6.Cg)([attr({attribute:"form"})],button_Button.prototype,"formId",void 0),(0,tslib_es6.Cg)([attr],button_Button.prototype,"formaction",void 0),(0,tslib_es6.Cg)([attr],button_Button.prototype,"formenctype",void 0),(0,tslib_es6.Cg)([attr],button_Button.prototype,"formmethod",void 0),(0,tslib_es6.Cg)([attr({mode:"boolean"})],button_Button.prototype,"formnovalidate",void 0),(0,tslib_es6.Cg)([attr],button_Button.prototype,"formtarget",void 0),(0,tslib_es6.Cg)([attr],button_Button.prototype,"type",void 0),(0,tslib_es6.Cg)([observable],button_Button.prototype,"defaultSlottedContent",void 0);class DelegatesARIAButton{}function node_observation_elements(e){return e?function(t,n,r){return 1===t.nodeType&&t.matches(e)}:function(e,t,n){return 1===e.nodeType}}(0,tslib_es6.Cg)([attr({attribute:"aria-expanded"})],DelegatesARIAButton.prototype,"ariaExpanded",void 0),(0,tslib_es6.Cg)([attr({attribute:"aria-pressed"})],DelegatesARIAButton.prototype,"ariaPressed",void 0),applyMixins(DelegatesARIAButton,ARIAGlobalStatesAndProperties),applyMixins(button_Button,StartEnd,DelegatesARIAButton);class NodeObservationBehavior{constructor(e,t){this.target=e,this.options=t,this.source=null}bind(e){const t=this.options.property;this.shouldUpdate=Observable.getAccessors(e).some((e=>e.name===t)),this.source=e,this.updateTarget(this.computeNodes()),this.shouldUpdate&&this.observe()}unbind(){this.updateTarget(emptyArray),this.source=null,this.shouldUpdate&&this.disconnect()}handleEvent(){this.updateTarget(this.computeNodes())}computeNodes(){let e=this.getNodes();return void 0!==this.options.filter&&(e=e.filter(this.options.filter)),e}updateTarget(e){this.source[this.options.property]=e}}class SlottedBehavior extends NodeObservationBehavior{constructor(e,t){super(e,t)}observe(){this.target.addEventListener("slotchange",this)}disconnect(){this.target.removeEventListener("slotchange",this)}getNodes(){return this.target.assignedNodes(this.options)}}function slotted(e){return"string"==typeof e&&(e={property:e}),new AttachedBehaviorHTMLDirective("fast-slotted",SlottedBehavior,e)}const buttonTemplate=(e,t)=>html`
    <button
        class="control"
        part="control"
        ?autofocus="${e=>e.autofocus}"
        ?disabled="${e=>e.disabled}"
        form="${e=>e.formId}"
        formaction="${e=>e.formaction}"
        formenctype="${e=>e.formenctype}"
        formmethod="${e=>e.formmethod}"
        formnovalidate="${e=>e.formnovalidate}"
        formtarget="${e=>e.formtarget}"
        name="${e=>e.name}"
        type="${e=>e.type}"
        value="${e=>e.value}"
        aria-atomic="${e=>e.ariaAtomic}"
        aria-busy="${e=>e.ariaBusy}"
        aria-controls="${e=>e.ariaControls}"
        aria-current="${e=>e.ariaCurrent}"
        aria-describedby="${e=>e.ariaDescribedby}"
        aria-details="${e=>e.ariaDetails}"
        aria-disabled="${e=>e.ariaDisabled}"
        aria-errormessage="${e=>e.ariaErrormessage}"
        aria-expanded="${e=>e.ariaExpanded}"
        aria-flowto="${e=>e.ariaFlowto}"
        aria-haspopup="${e=>e.ariaHaspopup}"
        aria-hidden="${e=>e.ariaHidden}"
        aria-invalid="${e=>e.ariaInvalid}"
        aria-keyshortcuts="${e=>e.ariaKeyshortcuts}"
        aria-label="${e=>e.ariaLabel}"
        aria-labelledby="${e=>e.ariaLabelledby}"
        aria-live="${e=>e.ariaLive}"
        aria-owns="${e=>e.ariaOwns}"
        aria-pressed="${e=>e.ariaPressed}"
        aria-relevant="${e=>e.ariaRelevant}"
        aria-roledescription="${e=>e.ariaRoledescription}"
        ${ref("control")}
    >
        ${startSlotTemplate(e,t)}
        <span class="content" part="content">
            <slot ${slotted("defaultSlottedContent")}></slot>
        </span>
        ${endSlotTemplate(e,t)}
    </button>
`;function can_use_dom_canUseDOM(){return!("undefined"==typeof window||!window.document||!window.document.createElement)}function dom_isHTMLElement(...e){return e.every((e=>e instanceof HTMLElement))}function getDisplayedNodes(e,t){if(e&&t&&dom_isHTMLElement(e))return Array.from(e.querySelectorAll(t)).filter((e=>null!==e.offsetParent))}function getKeyCode(e){return null===e?null:e.which||e.keyCode||e.charCode}function getNonce(){const e=document.querySelector('meta[property="csp-nonce"]');return e?e.getAttribute("content"):null}let _canUseFocusVisible,_canUseCssGrid;function canUseFocusVisible(){if("boolean"==typeof _canUseFocusVisible)return _canUseFocusVisible;if(!can_use_dom_canUseDOM())return _canUseFocusVisible=!1,_canUseFocusVisible;const e=document.createElement("style"),t=getNonce();null!==t&&e.setAttribute("nonce",t),document.head.appendChild(e);try{e.sheet.insertRule("foo:focus-visible {color:inherit}",0),_canUseFocusVisible=!0}catch(e){_canUseFocusVisible=!1}finally{document.head.removeChild(e)}return _canUseFocusVisible}function canUseCssGrid(){if("boolean"==typeof _canUseCssGrid)return _canUseCssGrid;try{_canUseCssGrid=CSS.supports("display","grid")}catch(e){_canUseCssGrid=!1}return _canUseCssGrid}function canUseForcedColors(){return canUseDOM()&&(window.matchMedia("(forced-colors: none)").matches||window.matchMedia("(forced-colors: active)").matches)}function resetDocumentCache(){_canUseCssGrid=void 0,_canUseFocusVisible=void 0}const canUsedForcedColors=null,focusVisible=canUseFocusVisible()?"focus-visible":"focus",disabledCursor="not-allowed",BaseButtonStyles=css_css`
	${display("inline-flex")} :host {
		outline: none;
		font-family: ${fontFamily};
		font-size: ${typeRampBaseFontSize};
		line-height: ${typeRampBaseLineHeight};
		color: ${buttonPrimaryForeground};
		background: ${buttonPrimaryBackground};
		border-radius: 2px;
		fill: currentColor;
		cursor: pointer;
	}
	.control {
		background: transparent;
		height: inherit;
		flex-grow: 1;
		box-sizing: border-box;
		display: inline-flex;
		justify-content: center;
		align-items: center;
		padding: ${buttonPaddingVertical} ${buttonPaddingHorizontal};
		white-space: wrap;
		outline: none;
		text-decoration: none;
		border: calc(${borderWidth} * 1px) solid ${buttonBorder};
		color: inherit;
		border-radius: inherit;
		fill: inherit;
		cursor: inherit;
		font-family: inherit;
	}
	:host(:hover) {
		background: ${buttonPrimaryHoverBackground};
	}
	:host(:active) {
		background: ${buttonPrimaryBackground};
	}
	.control:${focusVisible} {
		outline: calc(${borderWidth} * 1px) solid ${focusBorder};
		outline-offset: calc(${borderWidth} * 2px);
	}
	.control::-moz-focus-inner {
		border: 0;
	}
	:host([disabled]) {
		opacity: ${disabledOpacity};
		background: ${buttonPrimaryBackground};
		cursor: ${disabledCursor};
	}
	.content {
		display: flex;
	}
	.start {
		display: flex;
	}
	::slotted(svg),
	::slotted(span) {
		width: calc(${designUnit} * 4px);
		height: calc(${designUnit} * 4px);
	}
	.start {
		margin-inline-end: 8px;
	}
`,PrimaryButtonStyles=css_css`
	:host([appearance='primary']) {
		background: ${buttonPrimaryBackground};
		color: ${buttonPrimaryForeground};
	}
	:host([appearance='primary']:hover) {
		background: ${buttonPrimaryHoverBackground};
	}
	:host([appearance='primary']:active) .control:active {
		background: ${buttonPrimaryBackground};
	}
	:host([appearance='primary']) .control:${focusVisible} {
		outline: calc(${borderWidth} * 1px) solid ${focusBorder};
		outline-offset: calc(${borderWidth} * 2px);
	}
	:host([appearance='primary'][disabled]) {
		background: ${buttonPrimaryBackground};
	}
`,SecondaryButtonStyles=css_css`
	:host([appearance='secondary']) {
		background: ${buttonSecondaryBackground};
		color: ${buttonSecondaryForeground};
	}
	:host([appearance='secondary']:hover) {
		background: ${buttonSecondaryHoverBackground};
	}
	:host([appearance='secondary']:active) .control:active {
		background: ${buttonSecondaryBackground};
	}
	:host([appearance='secondary']) .control:${focusVisible} {
		outline: calc(${borderWidth} * 1px) solid ${focusBorder};
		outline-offset: calc(${borderWidth} * 2px);
	}
	:host([appearance='secondary'][disabled]) {
		background: ${buttonSecondaryBackground};
	}
`,IconButtonStyles=css_css`
	:host([appearance='icon']) {
		background: ${buttonIconBackground};
		border-radius: ${buttonIconCornerRadius};
		color: ${foreground};
	}
	:host([appearance='icon']:hover) {
		background: ${buttonIconHoverBackground};
		outline: 1px dotted ${contrastActiveBorder};
		outline-offset: -1px;
	}
	:host([appearance='icon']) .control {
		padding: ${buttonIconPadding};
		border: none;
	}
	:host([appearance='icon']:active) .control:active {
		background: ${buttonIconHoverBackground};
	}
	:host([appearance='icon']) .control:${focusVisible} {
		outline: calc(${borderWidth} * 1px) solid ${focusBorder};
		outline-offset: ${buttonIconFocusBorderOffset};
	}
	:host([appearance='icon'][disabled]) {
		background: ${buttonIconBackground};
	}
`,buttonStyles=(e,t)=>css_css`
	${BaseButtonStyles}
	${PrimaryButtonStyles}
	${SecondaryButtonStyles}
	${IconButtonStyles}
`;class Button extends button_Button{connectedCallback(){if(super.connectedCallback(),!this.appearance){const e=this.getAttribute("appearance");this.appearance=e}}attributeChangedCallback(e,t,n){"appearance"===e&&"icon"===n&&(this.getAttribute("aria-label")||(this.ariaLabel="Icon Button")),"aria-label"===e&&(this.ariaLabel=n),"disabled"===e&&(this.disabled=null!==n)}}(0,tslib_es6.Cg)([attr],Button.prototype,"appearance",void 0);const vsCodeButton=Button.compose({baseName:"button",template:buttonTemplate,styles:buttonStyles,shadowOptions:{delegatesFocus:!0}});class _Checkbox extends FoundationElement{}class FormAssociatedCheckbox extends(CheckableFormAssociated(_Checkbox)){constructor(){super(...arguments),this.proxy=document.createElement("input")}}class checkbox_Checkbox extends FormAssociatedCheckbox{constructor(){super(),this.initialValue="on",this.indeterminate=!1,this.keypressHandler=e=>{this.readOnly||e.key!==keySpace||(this.indeterminate&&(this.indeterminate=!1),this.checked=!this.checked)},this.clickHandler=e=>{this.disabled||this.readOnly||(this.indeterminate&&(this.indeterminate=!1),this.checked=!this.checked)},this.proxy.setAttribute("type","checkbox")}readOnlyChanged(){this.proxy instanceof HTMLInputElement&&(this.proxy.readOnly=this.readOnly)}}(0,tslib_es6.Cg)([attr({attribute:"readonly",mode:"boolean"})],checkbox_Checkbox.prototype,"readOnly",void 0),(0,tslib_es6.Cg)([observable],checkbox_Checkbox.prototype,"defaultSlottedNodes",void 0),(0,tslib_es6.Cg)([observable],checkbox_Checkbox.prototype,"indeterminate",void 0);const checkboxTemplate=(e,t)=>html`
    <template
        role="checkbox"
        aria-checked="${e=>e.checked}"
        aria-required="${e=>e.required}"
        aria-disabled="${e=>e.disabled}"
        aria-readonly="${e=>e.readOnly}"
        tabindex="${e=>e.disabled?null:0}"
        @keypress="${(e,t)=>e.keypressHandler(t.event)}"
        @click="${(e,t)=>e.clickHandler(t.event)}"
        class="${e=>e.readOnly?"readonly":""} ${e=>e.checked?"checked":""} ${e=>e.indeterminate?"indeterminate":""}"
    >
        <div part="control" class="control">
            <slot name="checked-indicator">
                ${t.checkedIndicator||""}
            </slot>
            <slot name="indeterminate-indicator">
                ${t.indeterminateIndicator||""}
            </slot>
        </div>
        <label
            part="label"
            class="${e=>e.defaultSlottedNodes&&e.defaultSlottedNodes.length?"label":"label label__hidden"}"
        >
            <slot ${slotted("defaultSlottedNodes")}></slot>
        </label>
    </template>
`,checkboxStyles=(e,t)=>css_css`
	${display("inline-flex")} :host {
		align-items: center;
		outline: none;
		margin: calc(${designUnit} * 1px) 0;
		user-select: none;
		font-size: ${typeRampBaseFontSize};
		line-height: ${typeRampBaseLineHeight};
	}
	.control {
		position: relative;
		width: calc(${designUnit} * 4px + 2px);
		height: calc(${designUnit} * 4px + 2px);
		box-sizing: border-box;
		border-radius: calc(${checkboxCornerRadius} * 1px);
		border: calc(${borderWidth} * 1px) solid ${checkboxBorder};
		background: ${checkboxBackground};
		outline: none;
		cursor: pointer;
	}
	.label {
		font-family: ${fontFamily};
		color: ${foreground};
		padding-inline-start: calc(${designUnit} * 2px + 2px);
		margin-inline-end: calc(${designUnit} * 2px + 2px);
		cursor: pointer;
	}
	.label__hidden {
		display: none;
		visibility: hidden;
	}
	.checked-indicator {
		width: 100%;
		height: 100%;
		display: block;
		fill: ${foreground};
		opacity: 0;
		pointer-events: none;
	}
	.indeterminate-indicator {
		border-radius: 2px;
		background: ${foreground};
		position: absolute;
		top: 50%;
		left: 50%;
		width: 50%;
		height: 50%;
		transform: translate(-50%, -50%);
		opacity: 0;
	}
	:host(:enabled) .control:hover {
		background: ${checkboxBackground};
		border-color: ${checkboxBorder};
	}
	:host(:enabled) .control:active {
		background: ${checkboxBackground};
		border-color: ${focusBorder};
	}
	:host(:${focusVisible}) .control {
		border: calc(${borderWidth} * 1px) solid ${focusBorder};
	}
	:host(.disabled) .label,
	:host(.readonly) .label,
	:host(.readonly) .control,
	:host(.disabled) .control {
		cursor: ${disabledCursor};
	}
	:host(.checked:not(.indeterminate)) .checked-indicator,
	:host(.indeterminate) .indeterminate-indicator {
		opacity: 1;
	}
	:host(.disabled) {
		opacity: ${disabledOpacity};
	}
`;class Checkbox extends checkbox_Checkbox{connectedCallback(){super.connectedCallback(),this.textContent?this.setAttribute("aria-label",this.textContent):this.setAttribute("aria-label","Checkbox")}}const vsCodeCheckbox=Checkbox.compose({baseName:"checkbox",template:checkboxTemplate,styles:checkboxStyles,checkedIndicator:'\n\t\t<svg \n\t\t\tpart="checked-indicator"\n\t\t\tclass="checked-indicator"\n\t\t\twidth="16" \n\t\t\theight="16" \n\t\t\tviewBox="0 0 16 16" \n\t\t\txmlns="http://www.w3.org/2000/svg" \n\t\t\tfill="currentColor"\n\t\t>\n\t\t\t<path \n\t\t\t\tfill-rule="evenodd" \n\t\t\t\tclip-rule="evenodd" \n\t\t\t\td="M14.431 3.323l-8.47 10-.79-.036-3.35-4.77.818-.574 2.978 4.24 8.051-9.506.764.646z"\n\t\t\t/>\n\t\t</svg>\n\t',indeterminateIndicator:'\n\t\t<div part="indeterminate-indicator" class="indeterminate-indicator"></div>\n\t'});function newSplice(e,t,n){return{index:e,removed:t,addedCount:n}}const EDIT_LEAVE=0,EDIT_UPDATE=1,EDIT_ADD=2,EDIT_DELETE=3;function calcEditDistances(e,t,n,r,o,i){const a=i-o+1,s=n-t+1,l=new Array(a);let c,u;for(let e=0;e<a;++e)l[e]=new Array(s),l[e][0]=e;for(let e=0;e<s;++e)l[0][e]=e;for(let n=1;n<a;++n)for(let i=1;i<s;++i)e[t+i-1]===r[o+n-1]?l[n][i]=l[n-1][i-1]:(c=l[n-1][i]+1,u=l[n][i-1]+1,l[n][i]=c<u?c:u);return l}function spliceOperationsFromEditDistances(e){let t=e.length-1,n=e[0].length-1,r=e[t][n];const o=[];for(;t>0||n>0;){if(0===t){o.push(EDIT_ADD),n--;continue}if(0===n){o.push(EDIT_DELETE),t--;continue}const i=e[t-1][n-1],a=e[t-1][n],s=e[t][n-1];let l;l=a<s?a<i?a:i:s<i?s:i,l===i?(i===r?o.push(EDIT_LEAVE):(o.push(EDIT_UPDATE),r=i),t--,n--):l===a?(o.push(EDIT_DELETE),t--,r=a):(o.push(EDIT_ADD),n--,r=s)}return o.reverse(),o}function sharedPrefix(e,t,n){for(let r=0;r<n;++r)if(e[r]!==t[r])return r;return n}function sharedSuffix(e,t,n){let r=e.length,o=t.length,i=0;for(;i<n&&e[--r]===t[--o];)i++;return i}function intersect(e,t,n,r){return t<n||r<e?-1:t===n||r===e?0:e<n?t<r?t-n:r-n:r<t?r-e:t-e}function calcSplices(e,t,n,r,o,i){let a=0,s=0;const l=Math.min(n-t,i-o);if(0===t&&0===o&&(a=sharedPrefix(e,r,l)),n===e.length&&i===r.length&&(s=sharedSuffix(e,r,l-a)),o+=a,i-=s,(n-=s)-(t+=a)==0&&i-o==0)return emptyArray;if(t===n){const e=newSplice(t,[],0);for(;o<i;)e.removed.push(r[o++]);return[e]}if(o===i)return[newSplice(t,[],n-t)];const c=spliceOperationsFromEditDistances(calcEditDistances(e,t,n,r,o,i)),u=[];let d,p=t,h=o;for(let e=0;e<c.length;++e)switch(c[e]){case EDIT_LEAVE:void 0!==d&&(u.push(d),d=void 0),p++,h++;break;case EDIT_UPDATE:void 0===d&&(d=newSplice(p,[],0)),d.addedCount++,p++,d.removed.push(r[h]),h++;break;case EDIT_ADD:void 0===d&&(d=newSplice(p,[],0)),d.addedCount++,p++;break;case EDIT_DELETE:void 0===d&&(d=newSplice(p,[],0)),d.removed.push(r[h]),h++}return void 0!==d&&u.push(d),u}const $push=Array.prototype.push;function mergeSplice(e,t,n,r){const o=newSplice(t,n,r);let i=!1,a=0;for(let t=0;t<e.length;t++){const n=e[t];if(n.index+=a,i)continue;const r=intersect(o.index,o.index+o.removed.length,n.index,n.index+n.addedCount);if(r>=0){e.splice(t,1),t--,a-=n.addedCount-n.removed.length,o.addedCount+=n.addedCount-r;const s=o.removed.length+n.removed.length-r;if(o.addedCount||s){let e=n.removed;if(o.index<n.index){const t=o.removed.slice(0,n.index-o.index);$push.apply(t,e),e=t}if(o.index+o.removed.length>n.index+n.addedCount){const t=o.removed.slice(n.index+n.addedCount-o.index);$push.apply(e,t)}o.removed=e,n.index<o.index&&(o.index=n.index)}else i=!0}else if(o.index<n.index){i=!0,e.splice(t,0,o),t++;const r=o.addedCount-o.removed.length;n.index+=r,a+=r}}i||e.push(o)}function createInitialSplices(e){const t=[];for(let n=0,r=e.length;n<r;n++){const r=e[n];mergeSplice(t,r.index,r.removed,r.addedCount)}return t}function projectArraySplices(e,t){let n=[];const r=createInitialSplices(t);for(let t=0,o=r.length;t<o;++t){const o=r[t];1!==o.addedCount||1!==o.removed.length?n=n.concat(calcSplices(e,o.index,o.index+o.addedCount,o.removed,0,o.removed.length)):o.removed[0]!==e[o.index]&&n.push(o)}return n}let arrayObservationEnabled=!1;function adjustIndex(e,t){let n=e.index;const r=t.length;return n>r?n=r-e.addedCount:n<0&&(n=r+e.removed.length+n-e.addedCount),n<0&&(n=0),e.index=n,e}class ArrayObserver extends SubscriberSet{constructor(e){super(e),this.oldCollection=void 0,this.splices=void 0,this.needsQueue=!0,this.call=this.flush,Reflect.defineProperty(e,"$fastController",{value:this,enumerable:!1})}subscribe(e){this.flush(),super.subscribe(e)}addSplice(e){void 0===this.splices?this.splices=[e]:this.splices.push(e),this.needsQueue&&(this.needsQueue=!1,DOM.queueUpdate(this))}reset(e){this.oldCollection=e,this.needsQueue&&(this.needsQueue=!1,DOM.queueUpdate(this))}flush(){const e=this.splices,t=this.oldCollection;if(void 0===e&&void 0===t)return;this.needsQueue=!0,this.splices=void 0,this.oldCollection=void 0;const n=void 0===t?projectArraySplices(this.source,e):calcSplices(this.source,0,this.source.length,t,0,t.length);this.notify(n)}}function enableArrayObservation(){if(arrayObservationEnabled)return;arrayObservationEnabled=!0,Observable.setArrayObserverFactory((e=>new ArrayObserver(e)));const e=Array.prototype;if(e.$fastPatch)return;Reflect.defineProperty(e,"$fastPatch",{value:1,enumerable:!1});const t=e.pop,n=e.push,r=e.reverse,o=e.shift,i=e.sort,a=e.splice,s=e.unshift;e.pop=function(){const e=this.length>0,n=t.apply(this,arguments),r=this.$fastController;return void 0!==r&&e&&r.addSplice(newSplice(this.length,[n],0)),n},e.push=function(){const e=n.apply(this,arguments),t=this.$fastController;return void 0!==t&&t.addSplice(adjustIndex(newSplice(this.length-arguments.length,[],arguments.length),this)),e},e.reverse=function(){let e;const t=this.$fastController;void 0!==t&&(t.flush(),e=this.slice());const n=r.apply(this,arguments);return void 0!==t&&t.reset(e),n},e.shift=function(){const e=this.length>0,t=o.apply(this,arguments),n=this.$fastController;return void 0!==n&&e&&n.addSplice(newSplice(0,[t],0)),t},e.sort=function(){let e;const t=this.$fastController;void 0!==t&&(t.flush(),e=this.slice());const n=i.apply(this,arguments);return void 0!==t&&t.reset(e),n},e.splice=function(){const e=a.apply(this,arguments),t=this.$fastController;return void 0!==t&&t.addSplice(adjustIndex(newSplice(+arguments[0],e,arguments.length>2?arguments.length-2:0),this)),e},e.unshift=function(){const e=s.apply(this,arguments),t=this.$fastController;return void 0!==t&&t.addSplice(adjustIndex(newSplice(0,[],arguments.length),this)),e}}const defaultRepeatOptions=Object.freeze({positioning:!1,recycle:!0});function bindWithoutPositioning(e,t,n,r){e.bind(t[n],r)}function bindWithPositioning(e,t,n,r){const o=Object.create(r);o.index=n,o.length=t.length,e.bind(t[n],o)}class RepeatBehavior{constructor(e,t,n,r,o,i){this.location=e,this.itemsBinding=t,this.templateBinding=r,this.options=i,this.source=null,this.views=[],this.items=null,this.itemsObserver=null,this.originalContext=void 0,this.childContext=void 0,this.bindView=bindWithoutPositioning,this.itemsBindingObserver=Observable.binding(t,this,n),this.templateBindingObserver=Observable.binding(r,this,o),i.positioning&&(this.bindView=bindWithPositioning)}bind(e,t){this.source=e,this.originalContext=t,this.childContext=Object.create(t),this.childContext.parent=e,this.childContext.parentContext=this.originalContext,this.items=this.itemsBindingObserver.observe(e,this.originalContext),this.template=this.templateBindingObserver.observe(e,this.originalContext),this.observeItems(!0),this.refreshAllViews()}unbind(){this.source=null,this.items=null,null!==this.itemsObserver&&this.itemsObserver.unsubscribe(this),this.unbindAllViews(),this.itemsBindingObserver.disconnect(),this.templateBindingObserver.disconnect()}handleChange(e,t){e===this.itemsBinding?(this.items=this.itemsBindingObserver.observe(this.source,this.originalContext),this.observeItems(),this.refreshAllViews()):e===this.templateBinding?(this.template=this.templateBindingObserver.observe(this.source,this.originalContext),this.refreshAllViews(!0)):this.updateViews(t)}observeItems(e=!1){if(!this.items)return void(this.items=emptyArray);const t=this.itemsObserver,n=this.itemsObserver=Observable.getNotifier(this.items),r=t!==n;r&&null!==t&&t.unsubscribe(this),(r||e)&&n.subscribe(this)}updateViews(e){const t=this.childContext,n=this.views,r=this.bindView,o=this.items,i=this.template,a=this.options.recycle,s=[];let l=0,c=0;for(let u=0,d=e.length;u<d;++u){const d=e[u],p=d.removed;let h=0,g=d.index;const m=g+d.addedCount,f=n.splice(d.index,p.length);for(c=s.length+f.length;g<m;++g){const e=n[g],u=e?e.firstChild:this.location;let d;a&&c>0?(h<=c&&f.length>0?(d=f[h],h++):(d=s[l],l++),c--):d=i.create(),n.splice(g,0,d),r(d,o,g,t),d.insertBefore(u)}f[h]&&s.push(...f.slice(h))}for(let e=l,t=s.length;e<t;++e)s[e].dispose();if(this.options.positioning)for(let e=0,t=n.length;e<t;++e){const r=n[e].context;r.length=t,r.index=e}}refreshAllViews(e=!1){const t=this.items,n=this.childContext,r=this.template,o=this.location,i=this.bindView;let a=t.length,s=this.views,l=s.length;if(0!==a&&!e&&this.options.recycle||(HTMLView.disposeContiguousBatch(s),l=0),0===l){this.views=s=new Array(a);for(let e=0;e<a;++e){const a=r.create();i(a,t,e,n),s[e]=a,a.insertBefore(o)}}else{let e=0;for(;e<a;++e)if(e<l)i(s[e],t,e,n);else{const a=r.create();i(a,t,e,n),s.push(a),a.insertBefore(o)}const c=s.splice(e,l-e);for(e=0,a=c.length;e<a;++e)c[e].dispose()}}unbindAllViews(){const e=this.views;for(let t=0,n=e.length;t<n;++t)e[t].unbind()}}class RepeatDirective extends HTMLDirective{constructor(e,t,n){super(),this.itemsBinding=e,this.templateBinding=t,this.options=n,this.createPlaceholder=DOM.createBlockPlaceholder,enableArrayObservation(),this.isItemsBindingVolatile=Observable.isVolatileBinding(e),this.isTemplateBindingVolatile=Observable.isVolatileBinding(t)}createBehavior(e){return new RepeatBehavior(e,this.itemsBinding,this.isItemsBindingVolatile,this.templateBinding,this.isTemplateBindingVolatile,this.options)}}function repeat(e,t,n=defaultRepeatOptions){return new RepeatDirective(e,"function"==typeof t?t:()=>t,Object.assign(Object.assign({},defaultRepeatOptions),n))}const eventAbort="abort",eventAfterPrint="afterprint",eventAnimationCancel="animationcancel",eventAnimationEnd="animationend",eventAnimationIteration="animationiteration",eventAnimationStart="animationstart",eventAppInstalled="appinstalled",eventBeforePrint="beforeprint",eventBeforeUnload="beforeunload",eventBeginEvent="beginEvent",eventBlocked="blocked",eventBlur="blur",eventCanPlay="canplay",eventCanPlayThrough="canplaythrough",eventChange="change",eventChargingChange="chargingchange",eventChargingTimeChange="chargingtimechange",eventClick="click",eventClose="close",eventComplete="complete",eventCompositionEnd="compositionend",eventCompositionStart="compositionstart",eventCompositionUpdate="compositionupdate",eventContextMenu="contextmenu",eventCopy="copy",eventCut="cut",eventDblClick="dblclick",eventDeviceChange="devicechange",eventDeviceMotion="devicemotion",eventDeviceOrientation="deviceorientation",eventDischargingTimeChange="dischargingtimechange",eventDrag="drag",eventDragEnd="dragend",eventDragEnter="dragenter",eventDragLeave="dragleave",eventDragOver="dragover",eventDragStart="dragstart",eventDrop="drop",eventDurationChange="durationchange",eventEmptied="emptied",eventEnded="ended",eventEndEvent="endevent",eventError="error",eventFocus="focus",eventFocusIn="focusin",eventFocusOut="focusout",eventFullScreenChange="fullscreenchange",eventFullScreenError="fullscreenerror",eventGamePadConnected="gamepadconnected",eventGamePadDisconnected="gamepaddisconnected",eventGotPointerCapture="gotpointercapture",eventHashChange="hashchange",eventLostPointerCapture="lostpointercapture",eventInput="input",eventInvalid="invalid",eventKeyDown="keydown",eventKeyUp="keyup",eventLevelChange="levelchange",eventLoad="load",eventLoadedData="loadeddata",eventLoadedMetaData="loadedmetadata",eventLoadEnd="loadend",eventLoadStart="loadstart",eventMessage="message",eventMessageError="messageerror",eventMouseDown="mousedown",eventMouseEnter="mouseenter",eventMouseLeave="mouseleave",eventMouseMove="mousemove",eventMouseOut="mouseout",eventMouseOver="mouseover",eventMouseUp="mouseup",eventNotificationClick="notificationclick",eventOffline="offline",eventOnline="online",eventOpen="open",eventOrientationChange="orientationchange",eventPageHide="pagehide",eventPageShow="pageshow",eventPaste="paste",eventPause="pause",eventPointerCancel="pointercancel",eventPointerDown="pointerdown",eventPointerEnter="pointerenter",eventPointerLeave="pointerleave",eventPointerLockChange="pointerlockchange",eventPointerLockError="pointerlockerror",eventPointerMove="pointermove",eventPointerOut="pointerout",eventPointerOver="pointerover",eventPointerUp="pointerup",eventPlay="play",eventPlaying="playing",eventPopState="popstate",eventProgress="progress",eventPush="push",eventPushSubscriptionChange="pushsubscriptionchange",eventRateChange="ratechange",eventReadyStateChange="readystatechange",eventRepeatEvent="repeatevent",eventReset="reset",eventResize="resize",eventResourceTimingBufferFull="resourcetimingbufferfull",eventScroll="scroll",eventSeeked="seeked",eventSeeking="seeking",eventSelect="select",eventShow="show",eventSlotChange="slotchange",eventStalled="stalled",eventStart="start",eventStorage="storage",eventSubmit="submit",eventSuccess="success",eventSuspend="suspend",eventSVGAbort="SVGAbort",eventSVGError="SVGError",eventSVGLoad="SVGLoad",eventSVGResize="SVGResize",eventSVGScroll="SVGScroll",eventSVGUnload="SVGUnload",eventSVGZoom="SVGZoom",eventTimeOut="timeout",eventTimeUpdate="timeupdate",eventTouchCancel="touchcancel",eventTouchEnd="touchend",eventTouchMove="touchmove",eventTouchStart="touchstart",eventTransitionEnd="transitionend",eventUnload="unload",eventUpgradeNeeded="upgradeneeded",eventUserProximity="userproximity",eventVersionChange="versionchange",eventVisibilityChange="visibilitychange",eventVolumeChange="volumechange",eventWaiting="waiting",eventWheel="wheel",GenerateHeaderOptions={none:"none",default:"default",sticky:"sticky"},DataGridCellTypes={default:"default",columnHeader:"columnheader",rowHeader:"rowheader"},DataGridRowTypes={default:"default",header:"header",stickyHeader:"sticky-header"};class data_grid_DataGrid extends FoundationElement{constructor(){super(),this.noTabbing=!1,this.generateHeader=GenerateHeaderOptions.default,this.rowsData=[],this.columnDefinitions=null,this.focusRowIndex=0,this.focusColumnIndex=0,this.rowsPlaceholder=null,this.generatedHeader=null,this.isUpdatingFocus=!1,this.pendingFocusUpdate=!1,this.rowindexUpdateQueued=!1,this.columnDefinitionsStale=!0,this.generatedGridTemplateColumns="",this.focusOnCell=(e,t,n)=>{if(0===this.rowElements.length)return this.focusRowIndex=0,void(this.focusColumnIndex=0);const r=Math.max(0,Math.min(this.rowElements.length-1,e)),o=this.rowElements[r].querySelectorAll('[role="cell"], [role="gridcell"], [role="columnheader"], [role="rowheader"]'),i=o[Math.max(0,Math.min(o.length-1,t))];n&&this.scrollHeight!==this.clientHeight&&(r<this.focusRowIndex&&this.scrollTop>0||r>this.focusRowIndex&&this.scrollTop<this.scrollHeight-this.clientHeight)&&i.scrollIntoView({block:"center",inline:"center"}),i.focus()},this.onChildListChange=(e,t)=>{e&&e.length&&(e.forEach((e=>{e.addedNodes.forEach((e=>{1===e.nodeType&&"row"===e.getAttribute("role")&&(e.columnDefinitions=this.columnDefinitions)}))})),this.queueRowIndexUpdate())},this.queueRowIndexUpdate=()=>{this.rowindexUpdateQueued||(this.rowindexUpdateQueued=!0,DOM.queueUpdate(this.updateRowIndexes))},this.updateRowIndexes=()=>{let e=this.gridTemplateColumns;if(void 0===e){if(""===this.generatedGridTemplateColumns&&this.rowElements.length>0){const e=this.rowElements[0];this.generatedGridTemplateColumns=new Array(e.cellElements.length).fill("1fr").join(" ")}e=this.generatedGridTemplateColumns}this.rowElements.forEach(((t,n)=>{const r=t;r.rowIndex=n,r.gridTemplateColumns=e,this.columnDefinitionsStale&&(r.columnDefinitions=this.columnDefinitions)})),this.rowindexUpdateQueued=!1,this.columnDefinitionsStale=!1}}static generateTemplateColumns(e){let t="";return e.forEach((e=>{t=`${t}${""===t?"":" "}1fr`})),t}noTabbingChanged(){this.$fastController.isConnected&&(this.noTabbing?this.setAttribute("tabIndex","-1"):this.setAttribute("tabIndex",this.contains(document.activeElement)||this===document.activeElement?"-1":"0"))}generateHeaderChanged(){this.$fastController.isConnected&&this.toggleGeneratedHeader()}gridTemplateColumnsChanged(){this.$fastController.isConnected&&this.updateRowIndexes()}rowsDataChanged(){null===this.columnDefinitions&&this.rowsData.length>0&&(this.columnDefinitions=data_grid_DataGrid.generateColumns(this.rowsData[0])),this.$fastController.isConnected&&this.toggleGeneratedHeader()}columnDefinitionsChanged(){null!==this.columnDefinitions?(this.generatedGridTemplateColumns=data_grid_DataGrid.generateTemplateColumns(this.columnDefinitions),this.$fastController.isConnected&&(this.columnDefinitionsStale=!0,this.queueRowIndexUpdate())):this.generatedGridTemplateColumns=""}headerCellItemTemplateChanged(){this.$fastController.isConnected&&null!==this.generatedHeader&&(this.generatedHeader.headerCellItemTemplate=this.headerCellItemTemplate)}focusRowIndexChanged(){this.$fastController.isConnected&&this.queueFocusUpdate()}focusColumnIndexChanged(){this.$fastController.isConnected&&this.queueFocusUpdate()}connectedCallback(){super.connectedCallback(),void 0===this.rowItemTemplate&&(this.rowItemTemplate=this.defaultRowItemTemplate),this.rowsPlaceholder=document.createComment(""),this.appendChild(this.rowsPlaceholder),this.toggleGeneratedHeader(),this.rowsRepeatBehavior=new RepeatDirective((e=>e.rowsData),(e=>e.rowItemTemplate),{positioning:!0}).createBehavior(this.rowsPlaceholder),this.$fastController.addBehaviors([this.rowsRepeatBehavior]),this.addEventListener("row-focused",this.handleRowFocus),this.addEventListener(eventFocus,this.handleFocus),this.addEventListener(eventKeyDown,this.handleKeydown),this.addEventListener(eventFocusOut,this.handleFocusOut),this.observer=new MutationObserver(this.onChildListChange),this.observer.observe(this,{childList:!0}),this.noTabbing&&this.setAttribute("tabindex","-1"),DOM.queueUpdate(this.queueRowIndexUpdate)}disconnectedCallback(){super.disconnectedCallback(),this.removeEventListener("row-focused",this.handleRowFocus),this.removeEventListener(eventFocus,this.handleFocus),this.removeEventListener(eventKeyDown,this.handleKeydown),this.removeEventListener(eventFocusOut,this.handleFocusOut),this.observer.disconnect(),this.rowsPlaceholder=null,this.generatedHeader=null}handleRowFocus(e){this.isUpdatingFocus=!0;const t=e.target;this.focusRowIndex=this.rowElements.indexOf(t),this.focusColumnIndex=t.focusColumnIndex,this.setAttribute("tabIndex","-1"),this.isUpdatingFocus=!1}handleFocus(e){this.focusOnCell(this.focusRowIndex,this.focusColumnIndex,!0)}handleFocusOut(e){null!==e.relatedTarget&&this.contains(e.relatedTarget)||this.setAttribute("tabIndex",this.noTabbing?"-1":"0")}handleKeydown(e){if(e.defaultPrevented)return;let t;const n=this.rowElements.length-1,r=this.offsetHeight+this.scrollTop,o=this.rowElements[n];switch(e.key){case keyArrowUp:e.preventDefault(),this.focusOnCell(this.focusRowIndex-1,this.focusColumnIndex,!0);break;case keyArrowDown:e.preventDefault(),this.focusOnCell(this.focusRowIndex+1,this.focusColumnIndex,!0);break;case keyPageUp:if(e.preventDefault(),0===this.rowElements.length){this.focusOnCell(0,0,!1);break}if(0===this.focusRowIndex)return void this.focusOnCell(0,this.focusColumnIndex,!1);for(t=this.focusRowIndex-1;t>=0;t--){const e=this.rowElements[t];if(e.offsetTop<this.scrollTop){this.scrollTop=e.offsetTop+e.clientHeight-this.clientHeight;break}}this.focusOnCell(t,this.focusColumnIndex,!1);break;case keyPageDown:if(e.preventDefault(),0===this.rowElements.length){this.focusOnCell(0,0,!1);break}if(this.focusRowIndex>=n||o.offsetTop+o.offsetHeight<=r)return void this.focusOnCell(n,this.focusColumnIndex,!1);for(t=this.focusRowIndex+1;t<=n;t++){const e=this.rowElements[t];if(e.offsetTop+e.offsetHeight>r){let t=0;this.generateHeader===GenerateHeaderOptions.sticky&&null!==this.generatedHeader&&(t=this.generatedHeader.clientHeight),this.scrollTop=e.offsetTop-t;break}}this.focusOnCell(t,this.focusColumnIndex,!1);break;case keyHome:e.ctrlKey&&(e.preventDefault(),this.focusOnCell(0,0,!0));break;case keyEnd:e.ctrlKey&&null!==this.columnDefinitions&&(e.preventDefault(),this.focusOnCell(this.rowElements.length-1,this.columnDefinitions.length-1,!0))}}queueFocusUpdate(){this.isUpdatingFocus&&(this.contains(document.activeElement)||this===document.activeElement)||!1===this.pendingFocusUpdate&&(this.pendingFocusUpdate=!0,DOM.queueUpdate((()=>this.updateFocus())))}updateFocus(){this.pendingFocusUpdate=!1,this.focusOnCell(this.focusRowIndex,this.focusColumnIndex,!0)}toggleGeneratedHeader(){if(null!==this.generatedHeader&&(this.removeChild(this.generatedHeader),this.generatedHeader=null),this.generateHeader!==GenerateHeaderOptions.none&&this.rowsData.length>0){const e=document.createElement(this.rowElementTag);return this.generatedHeader=e,this.generatedHeader.columnDefinitions=this.columnDefinitions,this.generatedHeader.gridTemplateColumns=this.gridTemplateColumns,this.generatedHeader.rowType=this.generateHeader===GenerateHeaderOptions.sticky?DataGridRowTypes.stickyHeader:DataGridRowTypes.header,void(null===this.firstChild&&null===this.rowsPlaceholder||this.insertBefore(e,null!==this.firstChild?this.firstChild:this.rowsPlaceholder))}}}data_grid_DataGrid.generateColumns=e=>Object.getOwnPropertyNames(e).map(((e,t)=>({columnDataKey:e,gridColumn:`${t}`}))),(0,tslib_es6.Cg)([attr({attribute:"no-tabbing",mode:"boolean"})],data_grid_DataGrid.prototype,"noTabbing",void 0),(0,tslib_es6.Cg)([attr({attribute:"generate-header"})],data_grid_DataGrid.prototype,"generateHeader",void 0),(0,tslib_es6.Cg)([attr({attribute:"grid-template-columns"})],data_grid_DataGrid.prototype,"gridTemplateColumns",void 0),(0,tslib_es6.Cg)([observable],data_grid_DataGrid.prototype,"rowsData",void 0),(0,tslib_es6.Cg)([observable],data_grid_DataGrid.prototype,"columnDefinitions",void 0),(0,tslib_es6.Cg)([observable],data_grid_DataGrid.prototype,"rowItemTemplate",void 0),(0,tslib_es6.Cg)([observable],data_grid_DataGrid.prototype,"cellItemTemplate",void 0),(0,tslib_es6.Cg)([observable],data_grid_DataGrid.prototype,"headerCellItemTemplate",void 0),(0,tslib_es6.Cg)([observable],data_grid_DataGrid.prototype,"focusRowIndex",void 0),(0,tslib_es6.Cg)([observable],data_grid_DataGrid.prototype,"focusColumnIndex",void 0),(0,tslib_es6.Cg)([observable],data_grid_DataGrid.prototype,"defaultRowItemTemplate",void 0),(0,tslib_es6.Cg)([observable],data_grid_DataGrid.prototype,"rowElementTag",void 0),(0,tslib_es6.Cg)([observable],data_grid_DataGrid.prototype,"rowElements",void 0);class ChildrenBehavior extends NodeObservationBehavior{constructor(e,t){super(e,t),this.observer=null,t.childList=!0}observe(){null===this.observer&&(this.observer=new MutationObserver(this.handleEvent.bind(this))),this.observer.observe(this.target,this.options)}disconnect(){this.observer.disconnect()}getNodes(){return"subtree"in this.options?Array.from(this.target.querySelectorAll(this.options.selector)):Array.from(this.target.childNodes)}}function children(e){return"string"==typeof e&&(e={property:e}),new AttachedBehaviorHTMLDirective("fast-children",ChildrenBehavior,e)}class data_grid_row_DataGridRow extends FoundationElement{constructor(){super(...arguments),this.rowType=DataGridRowTypes.default,this.rowData=null,this.columnDefinitions=null,this.isActiveRow=!1,this.cellsRepeatBehavior=null,this.cellsPlaceholder=null,this.focusColumnIndex=0,this.refocusOnLoad=!1,this.updateRowStyle=()=>{this.style.gridTemplateColumns=this.gridTemplateColumns}}gridTemplateColumnsChanged(){this.$fastController.isConnected&&this.updateRowStyle()}rowTypeChanged(){this.$fastController.isConnected&&this.updateItemTemplate()}rowDataChanged(){null!==this.rowData&&this.isActiveRow&&(this.refocusOnLoad=!0)}cellItemTemplateChanged(){this.updateItemTemplate()}headerCellItemTemplateChanged(){this.updateItemTemplate()}connectedCallback(){super.connectedCallback(),null===this.cellsRepeatBehavior&&(this.cellsPlaceholder=document.createComment(""),this.appendChild(this.cellsPlaceholder),this.updateItemTemplate(),this.cellsRepeatBehavior=new RepeatDirective((e=>e.columnDefinitions),(e=>e.activeCellItemTemplate),{positioning:!0}).createBehavior(this.cellsPlaceholder),this.$fastController.addBehaviors([this.cellsRepeatBehavior])),this.addEventListener("cell-focused",this.handleCellFocus),this.addEventListener(eventFocusOut,this.handleFocusout),this.addEventListener(eventKeyDown,this.handleKeydown),this.updateRowStyle(),this.refocusOnLoad&&(this.refocusOnLoad=!1,this.cellElements.length>this.focusColumnIndex&&this.cellElements[this.focusColumnIndex].focus())}disconnectedCallback(){super.disconnectedCallback(),this.removeEventListener("cell-focused",this.handleCellFocus),this.removeEventListener(eventFocusOut,this.handleFocusout),this.removeEventListener(eventKeyDown,this.handleKeydown)}handleFocusout(e){this.contains(e.target)||(this.isActiveRow=!1,this.focusColumnIndex=0)}handleCellFocus(e){this.isActiveRow=!0,this.focusColumnIndex=this.cellElements.indexOf(e.target),this.$emit("row-focused",this)}handleKeydown(e){if(e.defaultPrevented)return;let t=0;switch(e.key){case keyArrowLeft:t=Math.max(0,this.focusColumnIndex-1),this.cellElements[t].focus(),e.preventDefault();break;case keyArrowRight:t=Math.min(this.cellElements.length-1,this.focusColumnIndex+1),this.cellElements[t].focus(),e.preventDefault();break;case keyHome:e.ctrlKey||(this.cellElements[0].focus(),e.preventDefault());break;case keyEnd:e.ctrlKey||(this.cellElements[this.cellElements.length-1].focus(),e.preventDefault())}}updateItemTemplate(){this.activeCellItemTemplate=this.rowType===DataGridRowTypes.default&&void 0!==this.cellItemTemplate?this.cellItemTemplate:this.rowType===DataGridRowTypes.default&&void 0===this.cellItemTemplate?this.defaultCellItemTemplate:void 0!==this.headerCellItemTemplate?this.headerCellItemTemplate:this.defaultHeaderCellItemTemplate}}function createRowItemTemplate(e){const t=e.tagFor(data_grid_row_DataGridRow);return html`
    <${t}
        :rowData="${e=>e}"
        :cellItemTemplate="${(e,t)=>t.parent.cellItemTemplate}"
        :headerCellItemTemplate="${(e,t)=>t.parent.headerCellItemTemplate}"
    ></${t}>
`}(0,tslib_es6.Cg)([attr({attribute:"grid-template-columns"})],data_grid_row_DataGridRow.prototype,"gridTemplateColumns",void 0),(0,tslib_es6.Cg)([attr({attribute:"row-type"})],data_grid_row_DataGridRow.prototype,"rowType",void 0),(0,tslib_es6.Cg)([observable],data_grid_row_DataGridRow.prototype,"rowData",void 0),(0,tslib_es6.Cg)([observable],data_grid_row_DataGridRow.prototype,"columnDefinitions",void 0),(0,tslib_es6.Cg)([observable],data_grid_row_DataGridRow.prototype,"cellItemTemplate",void 0),(0,tslib_es6.Cg)([observable],data_grid_row_DataGridRow.prototype,"headerCellItemTemplate",void 0),(0,tslib_es6.Cg)([observable],data_grid_row_DataGridRow.prototype,"rowIndex",void 0),(0,tslib_es6.Cg)([observable],data_grid_row_DataGridRow.prototype,"isActiveRow",void 0),(0,tslib_es6.Cg)([observable],data_grid_row_DataGridRow.prototype,"activeCellItemTemplate",void 0),(0,tslib_es6.Cg)([observable],data_grid_row_DataGridRow.prototype,"defaultCellItemTemplate",void 0),(0,tslib_es6.Cg)([observable],data_grid_row_DataGridRow.prototype,"defaultHeaderCellItemTemplate",void 0),(0,tslib_es6.Cg)([observable],data_grid_row_DataGridRow.prototype,"cellElements",void 0);const dataGridTemplate=(e,t)=>{const n=createRowItemTemplate(e),r=e.tagFor(data_grid_row_DataGridRow);return html`
        <template
            role="grid"
            tabindex="0"
            :rowElementTag="${()=>r}"
            :defaultRowItemTemplate="${n}"
            ${children({property:"rowElements",filter:node_observation_elements("[role=row]")})}
        >
            <slot></slot>
        </template>
    `},defaultCellContentsTemplate=html`
    <template>
        ${e=>null===e.rowData||null===e.columnDefinition||null===e.columnDefinition.columnDataKey?null:e.rowData[e.columnDefinition.columnDataKey]}
    </template>
`,defaultHeaderCellContentsTemplate=html`
    <template>
        ${e=>null===e.columnDefinition?null:void 0===e.columnDefinition.title?e.columnDefinition.columnDataKey:e.columnDefinition.title}
    </template>
`;class data_grid_cell_DataGridCell extends FoundationElement{constructor(){super(...arguments),this.cellType=DataGridCellTypes.default,this.rowData=null,this.columnDefinition=null,this.isActiveCell=!1,this.customCellView=null,this.updateCellStyle=()=>{this.style.gridColumn=this.gridColumn}}cellTypeChanged(){this.$fastController.isConnected&&this.updateCellView()}gridColumnChanged(){this.$fastController.isConnected&&this.updateCellStyle()}columnDefinitionChanged(e,t){this.$fastController.isConnected&&this.updateCellView()}connectedCallback(){var e;super.connectedCallback(),this.addEventListener(eventFocusIn,this.handleFocusin),this.addEventListener(eventFocusOut,this.handleFocusout),this.addEventListener(eventKeyDown,this.handleKeydown),this.style.gridColumn=`${void 0===(null===(e=this.columnDefinition)||void 0===e?void 0:e.gridColumn)?0:this.columnDefinition.gridColumn}`,this.updateCellView(),this.updateCellStyle()}disconnectedCallback(){super.disconnectedCallback(),this.removeEventListener(eventFocusIn,this.handleFocusin),this.removeEventListener(eventFocusOut,this.handleFocusout),this.removeEventListener(eventKeyDown,this.handleKeydown),this.disconnectCellView()}handleFocusin(e){if(!this.isActiveCell){if(this.isActiveCell=!0,this.cellType===DataGridCellTypes.columnHeader){if(null!==this.columnDefinition&&!0!==this.columnDefinition.headerCellInternalFocusQueue&&"function"==typeof this.columnDefinition.headerCellFocusTargetCallback){const e=this.columnDefinition.headerCellFocusTargetCallback(this);null!==e&&e.focus()}}else if(null!==this.columnDefinition&&!0!==this.columnDefinition.cellInternalFocusQueue&&"function"==typeof this.columnDefinition.cellFocusTargetCallback){const e=this.columnDefinition.cellFocusTargetCallback(this);null!==e&&e.focus()}this.$emit("cell-focused",this)}}handleFocusout(e){this===document.activeElement||this.contains(document.activeElement)||(this.isActiveCell=!1)}handleKeydown(e){if(!(e.defaultPrevented||null===this.columnDefinition||this.cellType===DataGridCellTypes.default&&!0!==this.columnDefinition.cellInternalFocusQueue||this.cellType===DataGridCellTypes.columnHeader&&!0!==this.columnDefinition.headerCellInternalFocusQueue))switch(e.key){case keyEnter:case keyFunction2:if(this.contains(document.activeElement)&&document.activeElement!==this)return;if(this.cellType===DataGridCellTypes.columnHeader){if(void 0!==this.columnDefinition.headerCellFocusTargetCallback){const t=this.columnDefinition.headerCellFocusTargetCallback(this);null!==t&&t.focus(),e.preventDefault()}}else if(void 0!==this.columnDefinition.cellFocusTargetCallback){const t=this.columnDefinition.cellFocusTargetCallback(this);null!==t&&t.focus(),e.preventDefault()}break;case keyEscape:this.contains(document.activeElement)&&document.activeElement!==this&&(this.focus(),e.preventDefault())}}updateCellView(){if(this.disconnectCellView(),null!==this.columnDefinition)switch(this.cellType){case DataGridCellTypes.columnHeader:void 0!==this.columnDefinition.headerCellTemplate?this.customCellView=this.columnDefinition.headerCellTemplate.render(this,this):this.customCellView=defaultHeaderCellContentsTemplate.render(this,this);break;case void 0:case DataGridCellTypes.rowHeader:case DataGridCellTypes.default:void 0!==this.columnDefinition.cellTemplate?this.customCellView=this.columnDefinition.cellTemplate.render(this,this):this.customCellView=defaultCellContentsTemplate.render(this,this)}}disconnectCellView(){null!==this.customCellView&&(this.customCellView.dispose(),this.customCellView=null)}}function createCellItemTemplate(e){const t=e.tagFor(data_grid_cell_DataGridCell);return html`
    <${t}
        cell-type="${e=>e.isRowHeader?"rowheader":void 0}"
        grid-column="${(e,t)=>t.index+1}"
        :rowData="${(e,t)=>t.parent.rowData}"
        :columnDefinition="${e=>e}"
    ></${t}>
`}function createHeaderCellItemTemplate(e){const t=e.tagFor(data_grid_cell_DataGridCell);return html`
    <${t}
        cell-type="columnheader"
        grid-column="${(e,t)=>t.index+1}"
        :columnDefinition="${e=>e}"
    ></${t}>
`}(0,tslib_es6.Cg)([attr({attribute:"cell-type"})],data_grid_cell_DataGridCell.prototype,"cellType",void 0),(0,tslib_es6.Cg)([attr({attribute:"grid-column"})],data_grid_cell_DataGridCell.prototype,"gridColumn",void 0),(0,tslib_es6.Cg)([observable],data_grid_cell_DataGridCell.prototype,"rowData",void 0),(0,tslib_es6.Cg)([observable],data_grid_cell_DataGridCell.prototype,"columnDefinition",void 0);const dataGridRowTemplate=(e,t)=>html`
        <template
            role="row"
            class="${e=>"default"!==e.rowType?e.rowType:""}"
            :defaultCellItemTemplate="${createCellItemTemplate(e)}"
            :defaultHeaderCellItemTemplate="${createHeaderCellItemTemplate(e)}"
            ${children({property:"cellElements",filter:node_observation_elements('[role="cell"],[role="gridcell"],[role="columnheader"],[role="rowheader"]')})}
        >
            <slot ${slotted("slottedCellElements")}></slot>
        </template>
    `,dataGridCellTemplate=(e,t)=>html`
        <template
            tabindex="-1"
            role="${e=>e.cellType&&"default"!==e.cellType?e.cellType:"gridcell"}"
            class="
            ${e=>"columnheader"===e.cellType?"column-header":"rowheader"===e.cellType?"row-header":""}
            "
        >
            <slot></slot>
        </template>
    `,dataGridStyles=(e,t)=>css_css`
	:host {
		display: flex;
		position: relative;
		flex-direction: column;
		width: 100%;
	}
`,dataGridRowStyles=(e,t)=>css_css`
	:host {
		display: grid;
		padding: calc((${designUnit} / 4) * 1px) 0;
		box-sizing: border-box;
		width: 100%;
		background: transparent;
	}
	:host(.header) {
	}
	:host(.sticky-header) {
		background: ${background};
		position: sticky;
		top: 0;
	}
	:host(:hover) {
		background: ${listHoverBackground};
		outline: 1px dotted ${contrastActiveBorder};
		outline-offset: -1px;
	}
`,dataGridCellStyles=(e,t)=>css_css`
	:host {
		padding: calc(${designUnit} * 1px) calc(${designUnit} * 3px);
		color: ${foreground};
		opacity: 1;
		box-sizing: border-box;
		font-family: ${fontFamily};
		font-size: ${typeRampBaseFontSize};
		line-height: ${typeRampBaseLineHeight};
		font-weight: 400;
		border: solid calc(${borderWidth} * 1px) transparent;
		border-radius: calc(${cornerRadius} * 1px);
		white-space: wrap;
		overflow-wrap: anywhere;
	}
	:host(.column-header) {
		font-weight: 600;
		overflow-wrap: normal;
	}
	:host(:${focusVisible}),
	:host(:focus),
	:host(:active) {
		background: ${listActiveSelectionBackground};
		border: solid calc(${borderWidth} * 1px) ${focusBorder};
		color: ${listActiveSelectionForeground};
		outline: none;
	}
	:host(:${focusVisible}) ::slotted(*),
	:host(:focus) ::slotted(*),
	:host(:active) ::slotted(*) {
		color: ${listActiveSelectionForeground} !important;
	}
`;class dist_data_grid_DataGrid extends data_grid_DataGrid{connectedCallback(){super.connectedCallback(),this.getAttribute("aria-label")||this.setAttribute("aria-label","Data Grid")}}const vsCodeDataGrid=dist_data_grid_DataGrid.compose({baseName:"data-grid",baseClass:data_grid_DataGrid,template:dataGridTemplate,styles:dataGridStyles});class DataGridRow extends data_grid_row_DataGridRow{}const vsCodeDataGridRow=DataGridRow.compose({baseName:"data-grid-row",baseClass:data_grid_row_DataGridRow,template:dataGridRowTemplate,styles:dataGridRowStyles});class DataGridCell extends data_grid_cell_DataGridCell{}const vsCodeDataGridCell=DataGridCell.compose({baseName:"data-grid-cell",baseClass:data_grid_cell_DataGridCell,template:dataGridCellTemplate,styles:dataGridCellStyles}),Orientation={horizontal:"horizontal",vertical:"vertical"},DividerRole={separator:"separator",presentation:"presentation"};class divider_Divider extends FoundationElement{constructor(){super(...arguments),this.role=DividerRole.separator,this.orientation=Orientation.horizontal}}(0,tslib_es6.Cg)([attr],divider_Divider.prototype,"role",void 0),(0,tslib_es6.Cg)([attr],divider_Divider.prototype,"orientation",void 0);const dividerTemplate=(e,t)=>html`
    <template role="${e=>e.role}" aria-orientation="${e=>e.orientation}"></template>
`,dividerStyles=(e,t)=>css_css`
	${display("block")} :host {
		border: none;
		border-top: calc(${borderWidth} * 1px) solid ${dividerBackground};
		box-sizing: content-box;
		height: 0;
		margin: calc(${designUnit} * 1px) 0;
		width: 100%;
	}
`;class Divider extends divider_Divider{}const vsCodeDivider=Divider.compose({baseName:"divider",template:dividerTemplate,styles:dividerStyles});let uniqueIdCounter=0;function uniqueId(e=""){return`${e}${uniqueIdCounter++}`}function strings_format(e,...t){return e.replace(/{(\d+)}/g,(function(e,n){if(n>=t.length)return e;const r=t[n];return"number"==typeof r||r?r:""}))}function startsWith(e,t,n=0){return!(!e||!t)&&e.substr(n,t.length)===t}function isNullOrWhiteSpace(e){return!e||!e.trim()}function pascalCase(e){let t=`${e}`.replace(new RegExp(/[-_]+/,"g")," ").replace(new RegExp(/[^\w\s]/,"g"),"").replace(/^\s+|\s+$|\s+(?=\s)/g,"").replace(new RegExp(/\s+(.)(\w*)/,"g"),((e,t,n)=>`${t.toUpperCase()+n.toLowerCase()}`)).replace(new RegExp(/\w/),(e=>e.toUpperCase())),n=0;for(let e=0;e<t.length;e++){const r=t.charAt(e);if(r==r.toLowerCase()){n=e;break}}return n>1&&(t=`${t.charAt(0).toUpperCase()}${t.slice(1,n-1).toLowerCase()}`+t.slice(n-1)),t}function spinalCase(e){return`${e.charAt(0).toLowerCase()}${e.slice(1)}`.replace(/([A-Z]|[0-9])/g,(function(e,t){return`-${t.toLowerCase()}`}))}function findLastIndex(e,t){let n=e.length;for(;n--;)if(t(e[n],n,e))return n;return-1}function isListboxOption(e){return dom_isHTMLElement(e)&&("option"===e.getAttribute("role")||e instanceof HTMLOptionElement)}class ListboxOption extends FoundationElement{constructor(e,t,n,r){super(),this.defaultSelected=!1,this.dirtySelected=!1,this.selected=this.defaultSelected,this.dirtyValue=!1,e&&(this.textContent=e),t&&(this.initialValue=t),n&&(this.defaultSelected=n),r&&(this.selected=r),this.proxy=new Option(`${this.textContent}`,this.initialValue,this.defaultSelected,this.selected),this.proxy.disabled=this.disabled}checkedChanged(e,t){this.ariaChecked="boolean"!=typeof t?null:t?"true":"false"}contentChanged(e,t){this.proxy instanceof HTMLOptionElement&&(this.proxy.textContent=this.textContent),this.$emit("contentchange",null,{bubbles:!0})}defaultSelectedChanged(){this.dirtySelected||(this.selected=this.defaultSelected,this.proxy instanceof HTMLOptionElement&&(this.proxy.selected=this.defaultSelected))}disabledChanged(e,t){this.ariaDisabled=this.disabled?"true":"false",this.proxy instanceof HTMLOptionElement&&(this.proxy.disabled=this.disabled)}selectedAttributeChanged(){this.defaultSelected=this.selectedAttribute,this.proxy instanceof HTMLOptionElement&&(this.proxy.defaultSelected=this.defaultSelected)}selectedChanged(){this.ariaSelected=this.selected?"true":"false",this.dirtySelected||(this.dirtySelected=!0),this.proxy instanceof HTMLOptionElement&&(this.proxy.selected=this.selected)}initialValueChanged(e,t){this.dirtyValue||(this.value=this.initialValue,this.dirtyValue=!1)}get label(){var e;return null!==(e=this.value)&&void 0!==e?e:this.text}get text(){var e,t;return null!==(t=null===(e=this.textContent)||void 0===e?void 0:e.replace(/\s+/g," ").trim())&&void 0!==t?t:""}set value(e){const t=`${null!=e?e:""}`;this._value=t,this.dirtyValue=!0,this.proxy instanceof HTMLOptionElement&&(this.proxy.value=t),Observable.notify(this,"value")}get value(){var e;return Observable.track(this,"value"),null!==(e=this._value)&&void 0!==e?e:this.text}get form(){return this.proxy?this.proxy.form:null}}(0,tslib_es6.Cg)([observable],ListboxOption.prototype,"checked",void 0),(0,tslib_es6.Cg)([observable],ListboxOption.prototype,"content",void 0),(0,tslib_es6.Cg)([observable],ListboxOption.prototype,"defaultSelected",void 0),(0,tslib_es6.Cg)([attr({mode:"boolean"})],ListboxOption.prototype,"disabled",void 0),(0,tslib_es6.Cg)([attr({attribute:"selected",mode:"boolean"})],ListboxOption.prototype,"selectedAttribute",void 0),(0,tslib_es6.Cg)([observable],ListboxOption.prototype,"selected",void 0),(0,tslib_es6.Cg)([attr({attribute:"value",mode:"fromView"})],ListboxOption.prototype,"initialValue",void 0);class DelegatesARIAListboxOption{}(0,tslib_es6.Cg)([observable],DelegatesARIAListboxOption.prototype,"ariaChecked",void 0),(0,tslib_es6.Cg)([observable],DelegatesARIAListboxOption.prototype,"ariaPosInSet",void 0),(0,tslib_es6.Cg)([observable],DelegatesARIAListboxOption.prototype,"ariaSelected",void 0),(0,tslib_es6.Cg)([observable],DelegatesARIAListboxOption.prototype,"ariaSetSize",void 0),applyMixins(DelegatesARIAListboxOption,ARIAGlobalStatesAndProperties),applyMixins(ListboxOption,StartEnd,DelegatesARIAListboxOption);class Listbox extends FoundationElement{constructor(){super(...arguments),this._options=[],this.selectedIndex=-1,this.selectedOptions=[],this.shouldSkipFocus=!1,this.typeaheadBuffer="",this.typeaheadExpired=!0,this.typeaheadTimeout=-1}get firstSelectedOption(){var e;return null!==(e=this.selectedOptions[0])&&void 0!==e?e:null}get hasSelectableOptions(){return this.options.length>0&&!this.options.every((e=>e.disabled))}get length(){var e,t;return null!==(t=null===(e=this.options)||void 0===e?void 0:e.length)&&void 0!==t?t:0}get options(){return Observable.track(this,"options"),this._options}set options(e){this._options=e,Observable.notify(this,"options")}get typeAheadExpired(){return this.typeaheadExpired}set typeAheadExpired(e){this.typeaheadExpired=e}clickHandler(e){const t=e.target.closest("option,[role=option]");if(t&&!t.disabled)return this.selectedIndex=this.options.indexOf(t),!0}focusAndScrollOptionIntoView(e=this.firstSelectedOption){this.contains(document.activeElement)&&null!==e&&(e.focus(),requestAnimationFrame((()=>{e.scrollIntoView({block:"nearest"})})))}focusinHandler(e){this.shouldSkipFocus||e.target!==e.currentTarget||(this.setSelectedOptions(),this.focusAndScrollOptionIntoView()),this.shouldSkipFocus=!1}getTypeaheadMatches(){const e=this.typeaheadBuffer.replace(/[.*+\-?^${}()|[\]\\]/g,"\\$&"),t=new RegExp(`^${e}`,"gi");return this.options.filter((e=>e.text.trim().match(t)))}getSelectableIndex(e=this.selectedIndex,t){const n=e>t?-1:e<t?1:0,r=e+n;let o=null;switch(n){case-1:o=this.options.reduceRight(((e,t,n)=>!e&&!t.disabled&&n<r?t:e),o);break;case 1:o=this.options.reduce(((e,t,n)=>!e&&!t.disabled&&n>r?t:e),o)}return this.options.indexOf(o)}handleChange(e,t){"selected"===t&&(Listbox.slottedOptionFilter(e)&&(this.selectedIndex=this.options.indexOf(e)),this.setSelectedOptions())}handleTypeAhead(e){this.typeaheadTimeout&&window.clearTimeout(this.typeaheadTimeout),this.typeaheadTimeout=window.setTimeout((()=>this.typeaheadExpired=!0),Listbox.TYPE_AHEAD_TIMEOUT_MS),e.length>1||(this.typeaheadBuffer=`${this.typeaheadExpired?"":this.typeaheadBuffer}${e}`)}keydownHandler(e){if(this.disabled)return!0;this.shouldSkipFocus=!1;const t=e.key;switch(t){case keyHome:e.shiftKey||(e.preventDefault(),this.selectFirstOption());break;case keyArrowDown:e.shiftKey||(e.preventDefault(),this.selectNextOption());break;case keyArrowUp:e.shiftKey||(e.preventDefault(),this.selectPreviousOption());break;case keyEnd:e.preventDefault(),this.selectLastOption();break;case keyTab:return this.focusAndScrollOptionIntoView(),!0;case keyEnter:case keyEscape:return!0;case keySpace:if(this.typeaheadExpired)return!0;default:return 1===t.length&&this.handleTypeAhead(`${t}`),!0}}mousedownHandler(e){return this.shouldSkipFocus=!this.contains(document.activeElement),!0}multipleChanged(e,t){this.ariaMultiSelectable=t?"true":null}selectedIndexChanged(e,t){var n;if(this.hasSelectableOptions){if((null===(n=this.options[this.selectedIndex])||void 0===n?void 0:n.disabled)&&"number"==typeof e){const n=this.getSelectableIndex(e,t),r=n>-1?n:e;return this.selectedIndex=r,void(t===r&&this.selectedIndexChanged(t,r))}this.setSelectedOptions()}else this.selectedIndex=-1}selectedOptionsChanged(e,t){var n;const r=t.filter(Listbox.slottedOptionFilter);null===(n=this.options)||void 0===n||n.forEach((e=>{const t=Observable.getNotifier(e);t.unsubscribe(this,"selected"),e.selected=r.includes(e),t.subscribe(this,"selected")}))}selectFirstOption(){var e,t;this.disabled||(this.selectedIndex=null!==(t=null===(e=this.options)||void 0===e?void 0:e.findIndex((e=>!e.disabled)))&&void 0!==t?t:-1)}selectLastOption(){this.disabled||(this.selectedIndex=findLastIndex(this.options,(e=>!e.disabled)))}selectNextOption(){!this.disabled&&this.selectedIndex<this.options.length-1&&(this.selectedIndex+=1)}selectPreviousOption(){!this.disabled&&this.selectedIndex>0&&(this.selectedIndex=this.selectedIndex-1)}setDefaultSelectedOption(){var e,t;this.selectedIndex=null!==(t=null===(e=this.options)||void 0===e?void 0:e.findIndex((e=>e.defaultSelected)))&&void 0!==t?t:-1}setSelectedOptions(){var e,t,n;(null===(e=this.options)||void 0===e?void 0:e.length)&&(this.selectedOptions=[this.options[this.selectedIndex]],this.ariaActiveDescendant=null!==(n=null===(t=this.firstSelectedOption)||void 0===t?void 0:t.id)&&void 0!==n?n:"",this.focusAndScrollOptionIntoView())}slottedOptionsChanged(e,t){this.options=t.reduce(((e,t)=>(isListboxOption(t)&&e.push(t),e)),[]);const n=`${this.options.length}`;this.options.forEach(((e,t)=>{e.id||(e.id=uniqueId("option-")),e.ariaPosInSet=`${t+1}`,e.ariaSetSize=n})),this.$fastController.isConnected&&(this.setSelectedOptions(),this.setDefaultSelectedOption())}typeaheadBufferChanged(e,t){if(this.$fastController.isConnected){const e=this.getTypeaheadMatches();if(e.length){const t=this.options.indexOf(e[0]);t>-1&&(this.selectedIndex=t)}this.typeaheadExpired=!1}}}Listbox.slottedOptionFilter=e=>isListboxOption(e)&&!e.hidden,Listbox.TYPE_AHEAD_TIMEOUT_MS=1e3,(0,tslib_es6.Cg)([attr({mode:"boolean"})],Listbox.prototype,"disabled",void 0),(0,tslib_es6.Cg)([observable],Listbox.prototype,"selectedIndex",void 0),(0,tslib_es6.Cg)([observable],Listbox.prototype,"selectedOptions",void 0),(0,tslib_es6.Cg)([observable],Listbox.prototype,"slottedOptions",void 0),(0,tslib_es6.Cg)([observable],Listbox.prototype,"typeaheadBuffer",void 0);class DelegatesARIAListbox{}function wrapInBounds(e,t,n){return n<e?t:n>t?e:n}function limit(e,t,n){return Math.min(Math.max(n,e),t)}function inRange(e,t,n=0){return[t,n]=[t,n].sort(((e,t)=>e-t)),t<=e&&e<n}(0,tslib_es6.Cg)([observable],DelegatesARIAListbox.prototype,"ariaActiveDescendant",void 0),(0,tslib_es6.Cg)([observable],DelegatesARIAListbox.prototype,"ariaDisabled",void 0),(0,tslib_es6.Cg)([observable],DelegatesARIAListbox.prototype,"ariaExpanded",void 0),(0,tslib_es6.Cg)([observable],DelegatesARIAListbox.prototype,"ariaMultiSelectable",void 0),applyMixins(DelegatesARIAListbox,ARIAGlobalStatesAndProperties),applyMixins(Listbox,DelegatesARIAListbox);class ListboxElement extends Listbox{constructor(){super(...arguments),this.activeIndex=-1,this.rangeStartIndex=-1}get activeOption(){return this.options[this.activeIndex]}get checkedOptions(){var e;return null===(e=this.options)||void 0===e?void 0:e.filter((e=>e.checked))}get firstSelectedOptionIndex(){return this.options.indexOf(this.firstSelectedOption)}activeIndexChanged(e,t){var n,r;this.ariaActiveDescendant=null!==(r=null===(n=this.options[t])||void 0===n?void 0:n.id)&&void 0!==r?r:"",this.focusAndScrollOptionIntoView()}checkActiveIndex(){if(!this.multiple)return;const e=this.activeOption;e&&(e.checked=!0)}checkFirstOption(e=!1){e?(-1===this.rangeStartIndex&&(this.rangeStartIndex=this.activeIndex+1),this.options.forEach(((e,t)=>{e.checked=inRange(t,this.rangeStartIndex)}))):this.uncheckAllOptions(),this.activeIndex=0,this.checkActiveIndex()}checkLastOption(e=!1){e?(-1===this.rangeStartIndex&&(this.rangeStartIndex=this.activeIndex),this.options.forEach(((e,t)=>{e.checked=inRange(t,this.rangeStartIndex,this.options.length)}))):this.uncheckAllOptions(),this.activeIndex=this.options.length-1,this.checkActiveIndex()}connectedCallback(){super.connectedCallback(),this.addEventListener("focusout",this.focusoutHandler)}disconnectedCallback(){this.removeEventListener("focusout",this.focusoutHandler),super.disconnectedCallback()}checkNextOption(e=!1){e?(-1===this.rangeStartIndex&&(this.rangeStartIndex=this.activeIndex),this.options.forEach(((e,t)=>{e.checked=inRange(t,this.rangeStartIndex,this.activeIndex+1)}))):this.uncheckAllOptions(),this.activeIndex+=this.activeIndex<this.options.length-1?1:0,this.checkActiveIndex()}checkPreviousOption(e=!1){e?(-1===this.rangeStartIndex&&(this.rangeStartIndex=this.activeIndex),1===this.checkedOptions.length&&(this.rangeStartIndex+=1),this.options.forEach(((e,t)=>{e.checked=inRange(t,this.activeIndex,this.rangeStartIndex)}))):this.uncheckAllOptions(),this.activeIndex-=this.activeIndex>0?1:0,this.checkActiveIndex()}clickHandler(e){var t;if(!this.multiple)return super.clickHandler(e);const n=null===(t=e.target)||void 0===t?void 0:t.closest("[role=option]");return n&&!n.disabled?(this.uncheckAllOptions(),this.activeIndex=this.options.indexOf(n),this.checkActiveIndex(),this.toggleSelectedForAllCheckedOptions(),!0):void 0}focusAndScrollOptionIntoView(){super.focusAndScrollOptionIntoView(this.activeOption)}focusinHandler(e){if(!this.multiple)return super.focusinHandler(e);this.shouldSkipFocus||e.target!==e.currentTarget||(this.uncheckAllOptions(),-1===this.activeIndex&&(this.activeIndex=-1!==this.firstSelectedOptionIndex?this.firstSelectedOptionIndex:0),this.checkActiveIndex(),this.setSelectedOptions(),this.focusAndScrollOptionIntoView()),this.shouldSkipFocus=!1}focusoutHandler(e){this.multiple&&this.uncheckAllOptions()}keydownHandler(e){if(!this.multiple)return super.keydownHandler(e);if(this.disabled)return!0;const{key:t,shiftKey:n}=e;switch(this.shouldSkipFocus=!1,t){case keyHome:return void this.checkFirstOption(n);case keyArrowDown:return void this.checkNextOption(n);case keyArrowUp:return void this.checkPreviousOption(n);case keyEnd:return void this.checkLastOption(n);case keyTab:return this.focusAndScrollOptionIntoView(),!0;case keyEscape:return this.uncheckAllOptions(),this.checkActiveIndex(),!0;case keySpace:if(e.preventDefault(),this.typeAheadExpired)return void this.toggleSelectedForAllCheckedOptions();default:return 1===t.length&&this.handleTypeAhead(`${t}`),!0}}mousedownHandler(e){if(e.offsetX>=0&&e.offsetX<=this.scrollWidth)return super.mousedownHandler(e)}multipleChanged(e,t){var n;this.ariaMultiSelectable=t?"true":null,null===(n=this.options)||void 0===n||n.forEach((e=>{e.checked=!t&&void 0})),this.setSelectedOptions()}setSelectedOptions(){this.multiple?this.$fastController.isConnected&&this.options&&(this.selectedOptions=this.options.filter((e=>e.selected)),this.focusAndScrollOptionIntoView()):super.setSelectedOptions()}sizeChanged(e,t){var n;const r=Math.max(0,parseInt(null!==(n=null==t?void 0:t.toFixed())&&void 0!==n?n:"",10));r!==t&&DOM.queueUpdate((()=>{this.size=r}))}toggleSelectedForAllCheckedOptions(){const e=this.checkedOptions.filter((e=>!e.disabled)),t=!e.every((e=>e.selected));e.forEach((e=>e.selected=t)),this.selectedIndex=this.options.indexOf(e[e.length-1]),this.setSelectedOptions()}typeaheadBufferChanged(e,t){if(this.multiple){if(this.$fastController.isConnected){const e=this.getTypeaheadMatches(),t=this.options.indexOf(e[0]);t>-1&&(this.activeIndex=t,this.uncheckAllOptions(),this.checkActiveIndex()),this.typeAheadExpired=!1}}else super.typeaheadBufferChanged(e,t)}uncheckAllOptions(e=!1){this.options.forEach((e=>e.checked=!this.multiple&&void 0)),e||(this.rangeStartIndex=-1)}}(0,tslib_es6.Cg)([observable],ListboxElement.prototype,"activeIndex",void 0),(0,tslib_es6.Cg)([attr({mode:"boolean"})],ListboxElement.prototype,"multiple",void 0),(0,tslib_es6.Cg)([attr({converter:nullableNumberConverter})],ListboxElement.prototype,"size",void 0);class _Select extends ListboxElement{}class FormAssociatedSelect extends(FormAssociated(_Select)){constructor(){super(...arguments),this.proxy=document.createElement("select")}}const SelectPosition={above:"above",below:"below"};class Select extends FormAssociatedSelect{constructor(){super(...arguments),this.open=!1,this.forcedPosition=!1,this.listboxId=uniqueId("listbox-"),this.maxHeight=0}openChanged(e,t){if(this.collapsible){if(this.open)return this.ariaControls=this.listboxId,this.ariaExpanded="true",this.setPositioning(),this.focusAndScrollOptionIntoView(),this.indexWhenOpened=this.selectedIndex,void DOM.queueUpdate((()=>this.focus()));this.ariaControls="",this.ariaExpanded="false"}}get collapsible(){return!(this.multiple||"number"==typeof this.size)}get value(){return Observable.track(this,"value"),this._value}set value(e){var t,n,r,o,i,a,s;const l=`${this._value}`;if(null===(t=this._options)||void 0===t?void 0:t.length){const t=this._options.findIndex((t=>t.value===e)),l=null!==(r=null===(n=this._options[this.selectedIndex])||void 0===n?void 0:n.value)&&void 0!==r?r:null,c=null!==(i=null===(o=this._options[t])||void 0===o?void 0:o.value)&&void 0!==i?i:null;-1!==t&&l===c||(e="",this.selectedIndex=t),e=null!==(s=null===(a=this.firstSelectedOption)||void 0===a?void 0:a.value)&&void 0!==s?s:e}l!==e&&(this._value=e,super.valueChanged(l,e),Observable.notify(this,"value"),this.updateDisplayValue())}updateValue(e){var t,n;this.$fastController.isConnected&&(this.value=null!==(n=null===(t=this.firstSelectedOption)||void 0===t?void 0:t.value)&&void 0!==n?n:""),e&&(this.$emit("input"),this.$emit("change",this,{bubbles:!0,composed:void 0}))}selectedIndexChanged(e,t){super.selectedIndexChanged(e,t),this.updateValue()}positionChanged(e,t){this.positionAttribute=t,this.setPositioning()}setPositioning(){const e=this.getBoundingClientRect(),t=window.innerHeight-e.bottom;this.position=this.forcedPosition?this.positionAttribute:e.top>t?SelectPosition.above:SelectPosition.below,this.positionAttribute=this.forcedPosition?this.positionAttribute:this.position,this.maxHeight=this.position===SelectPosition.above?~~e.top:~~t}get displayValue(){var e,t;return Observable.track(this,"displayValue"),null!==(t=null===(e=this.firstSelectedOption)||void 0===e?void 0:e.text)&&void 0!==t?t:""}disabledChanged(e,t){super.disabledChanged&&super.disabledChanged(e,t),this.ariaDisabled=this.disabled?"true":"false"}formResetCallback(){this.setProxyOptions(),super.setDefaultSelectedOption(),-1===this.selectedIndex&&(this.selectedIndex=0)}clickHandler(e){if(!this.disabled){if(this.open){const t=e.target.closest("option,[role=option]");if(t&&t.disabled)return}return super.clickHandler(e),this.open=this.collapsible&&!this.open,this.open||this.indexWhenOpened===this.selectedIndex||this.updateValue(!0),!0}}focusoutHandler(e){var t;if(super.focusoutHandler(e),!this.open)return!0;const n=e.relatedTarget;this.isSameNode(n)?this.focus():(null===(t=this.options)||void 0===t?void 0:t.includes(n))||(this.open=!1,this.indexWhenOpened!==this.selectedIndex&&this.updateValue(!0))}handleChange(e,t){super.handleChange(e,t),"value"===t&&this.updateValue()}slottedOptionsChanged(e,t){this.options.forEach((e=>{Observable.getNotifier(e).unsubscribe(this,"value")})),super.slottedOptionsChanged(e,t),this.options.forEach((e=>{Observable.getNotifier(e).subscribe(this,"value")})),this.setProxyOptions(),this.updateValue()}mousedownHandler(e){var t;return e.offsetX>=0&&e.offsetX<=(null===(t=this.listbox)||void 0===t?void 0:t.scrollWidth)?super.mousedownHandler(e):this.collapsible}multipleChanged(e,t){super.multipleChanged(e,t),this.proxy&&(this.proxy.multiple=t)}selectedOptionsChanged(e,t){var n;super.selectedOptionsChanged(e,t),null===(n=this.options)||void 0===n||n.forEach(((e,t)=>{var n;const r=null===(n=this.proxy)||void 0===n?void 0:n.options.item(t);r&&(r.selected=e.selected)}))}setDefaultSelectedOption(){var e;const t=null!==(e=this.options)&&void 0!==e?e:Array.from(this.children).filter(Listbox.slottedOptionFilter),n=null==t?void 0:t.findIndex((e=>e.hasAttribute("selected")||e.selected||e.value===this.value));this.selectedIndex=-1===n?0:n}setProxyOptions(){this.proxy instanceof HTMLSelectElement&&this.options&&(this.proxy.options.length=0,this.options.forEach((e=>{const t=e.proxy||(e instanceof HTMLOptionElement?e.cloneNode():null);t&&this.proxy.options.add(t)})))}keydownHandler(e){super.keydownHandler(e);const t=e.key||e.key.charCodeAt(0);switch(t){case keySpace:e.preventDefault(),this.collapsible&&this.typeAheadExpired&&(this.open=!this.open);break;case keyHome:case keyEnd:e.preventDefault();break;case keyEnter:e.preventDefault(),this.open=!this.open;break;case keyEscape:this.collapsible&&this.open&&(e.preventDefault(),this.open=!1);break;case keyTab:return this.collapsible&&this.open&&(e.preventDefault(),this.open=!1),!0}return this.open||this.indexWhenOpened===this.selectedIndex||(this.updateValue(!0),this.indexWhenOpened=this.selectedIndex),!(t===keyArrowDown||t===keyArrowUp)}connectedCallback(){super.connectedCallback(),this.forcedPosition=!!this.positionAttribute,this.addEventListener("contentchange",this.updateDisplayValue)}disconnectedCallback(){this.removeEventListener("contentchange",this.updateDisplayValue),super.disconnectedCallback()}sizeChanged(e,t){super.sizeChanged(e,t),this.proxy&&(this.proxy.size=t)}updateDisplayValue(){this.collapsible&&Observable.notify(this,"displayValue")}}(0,tslib_es6.Cg)([attr({attribute:"open",mode:"boolean"})],Select.prototype,"open",void 0),(0,tslib_es6.Cg)([observable_volatile],Select.prototype,"collapsible",null),(0,tslib_es6.Cg)([observable],Select.prototype,"control",void 0),(0,tslib_es6.Cg)([attr({attribute:"position"})],Select.prototype,"positionAttribute",void 0),(0,tslib_es6.Cg)([observable],Select.prototype,"position",void 0),(0,tslib_es6.Cg)([observable],Select.prototype,"maxHeight",void 0);class DelegatesARIASelect{}function when(e,t){const n="function"==typeof t?t:()=>t;return(t,r)=>e(t,r)?n(t,r):null}(0,tslib_es6.Cg)([observable],DelegatesARIASelect.prototype,"ariaControls",void 0),applyMixins(DelegatesARIASelect,DelegatesARIAListbox),applyMixins(Select,StartEnd,DelegatesARIASelect);const selectTemplate=(e,t)=>html`
    <template
        class="${e=>[e.collapsible&&"collapsible",e.collapsible&&e.open&&"open",e.disabled&&"disabled",e.collapsible&&e.position].filter(Boolean).join(" ")}"
        aria-activedescendant="${e=>e.ariaActiveDescendant}"
        aria-controls="${e=>e.ariaControls}"
        aria-disabled="${e=>e.ariaDisabled}"
        aria-expanded="${e=>e.ariaExpanded}"
        aria-haspopup="${e=>e.collapsible?"listbox":null}"
        aria-multiselectable="${e=>e.ariaMultiSelectable}"
        ?open="${e=>e.open}"
        role="combobox"
        tabindex="${e=>e.disabled?null:"0"}"
        @click="${(e,t)=>e.clickHandler(t.event)}"
        @focusin="${(e,t)=>e.focusinHandler(t.event)}"
        @focusout="${(e,t)=>e.focusoutHandler(t.event)}"
        @keydown="${(e,t)=>e.keydownHandler(t.event)}"
        @mousedown="${(e,t)=>e.mousedownHandler(t.event)}"
    >
        ${when((e=>e.collapsible),html`
                <div
                    class="control"
                    part="control"
                    ?disabled="${e=>e.disabled}"
                    ${ref("control")}
                >
                    ${startSlotTemplate(e,t)}
                    <slot name="button-container">
                        <div class="selected-value" part="selected-value">
                            <slot name="selected-value">${e=>e.displayValue}</slot>
                        </div>
                        <div aria-hidden="true" class="indicator" part="indicator">
                            <slot name="indicator">
                                ${t.indicator||""}
                            </slot>
                        </div>
                    </slot>
                    ${endSlotTemplate(e,t)}
                </div>
            `)}
        <div
            class="listbox"
            id="${e=>e.listboxId}"
            part="listbox"
            role="listbox"
            ?disabled="${e=>e.disabled}"
            ?hidden="${e=>!!e.collapsible&&!e.open}"
            ${ref("listbox")}
        >
            <slot
                ${slotted({filter:Listbox.slottedOptionFilter,flatten:!0,property:"slottedOptions"})}
            ></slot>
        </div>
    </template>
`,dropdownStyles=(e,t)=>css_css`
	${display("inline-flex")} :host {
		background: ${dropdownBackground};
		box-sizing: border-box;
		color: ${foreground};
		contain: contents;
		font-family: ${fontFamily};
		height: calc(${inputHeight} * 1px);
		position: relative;
		user-select: none;
		min-width: ${inputMinWidth};
		outline: none;
		vertical-align: top;
	}
	.control {
		align-items: center;
		box-sizing: border-box;
		border: calc(${borderWidth} * 1px) solid ${dropdownBorder};
		border-radius: calc(${cornerRadius} * 1px);
		cursor: pointer;
		display: flex;
		font-family: inherit;
		font-size: ${typeRampBaseFontSize};
		line-height: ${typeRampBaseLineHeight};
		min-height: 100%;
		padding: 2px 6px 2px 8px;
		width: 100%;
	}
	.listbox {
		background: ${dropdownBackground};
		border: calc(${borderWidth} * 1px) solid ${focusBorder};
		border-radius: calc(${cornerRadius} * 1px);
		box-sizing: border-box;
		display: inline-flex;
		flex-direction: column;
		left: 0;
		max-height: ${dropdownListMaxHeight};
		padding: 0 0 calc(${designUnit} * 1px) 0;
		overflow-y: auto;
		position: absolute;
		width: 100%;
		z-index: 1;
	}
	.listbox[hidden] {
		display: none;
	}
	:host(:${focusVisible}) .control {
		border-color: ${focusBorder};
	}
	:host(:not([disabled]):hover) {
		background: ${dropdownBackground};
		border-color: ${dropdownBorder};
	}
	:host(:${focusVisible}) ::slotted([aria-selected="true"][role="option"]:not([disabled])) {
		background: ${listActiveSelectionBackground};
		border: calc(${borderWidth} * 1px) solid ${focusBorder};
		color: ${listActiveSelectionForeground};
	}
	:host([disabled]) {
		cursor: ${disabledCursor};
		opacity: ${disabledOpacity};
	}
	:host([disabled]) .control {
		cursor: ${disabledCursor};
		user-select: none;
	}
	:host([disabled]:hover) {
		background: ${dropdownBackground};
		color: ${foreground};
		fill: currentcolor;
	}
	:host(:not([disabled])) .control:active {
		border-color: ${focusBorder};
	}
	:host(:empty) .listbox {
		display: none;
	}
	:host([open]) .control {
		border-color: ${focusBorder};
	}
	:host([open][position='above']) .listbox,
	:host([open][position='below']) .control {
		border-bottom-left-radius: 0;
		border-bottom-right-radius: 0;
	}
	:host([open][position='above']) .control,
	:host([open][position='below']) .listbox {
		border-top-left-radius: 0;
		border-top-right-radius: 0;
	}
	:host([open][position='above']) .listbox {
		bottom: calc(${inputHeight} * 1px);
	}
	:host([open][position='below']) .listbox {
		top: calc(${inputHeight} * 1px);
	}
	.selected-value {
		flex: 1 1 auto;
		font-family: inherit;
		overflow: hidden;
		text-align: start;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.indicator {
		flex: 0 0 auto;
		margin-inline-start: 1em;
	}
	slot[name='listbox'] {
		display: none;
		width: 100%;
	}
	:host([open]) slot[name='listbox'] {
		display: flex;
		position: absolute;
	}
	.end {
		margin-inline-start: auto;
	}
	.start,
	.end,
	.indicator,
	.select-indicator,
	::slotted(svg),
	::slotted(span) {
		fill: currentcolor;
		height: 1em;
		min-height: calc(${designUnit} * 4px);
		min-width: calc(${designUnit} * 4px);
		width: 1em;
	}
	::slotted([role='option']),
	::slotted(option) {
		flex: 0 0 auto;
	}
`;class Dropdown extends Select{}const vsCodeDropdown=Dropdown.compose({baseName:"dropdown",template:selectTemplate,styles:dropdownStyles,indicator:'\n\t\t<svg \n\t\t\tclass="select-indicator"\n\t\t\tpart="select-indicator"\n\t\t\twidth="16" \n\t\t\theight="16" \n\t\t\tviewBox="0 0 16 16" \n\t\t\txmlns="http://www.w3.org/2000/svg" \n\t\t\tfill="currentColor"\n\t\t>\n\t\t\t<path \n\t\t\t\tfill-rule="evenodd" \n\t\t\t\tclip-rule="evenodd" \n\t\t\t\td="M7.976 10.072l4.357-4.357.62.618L8.284 11h-.618L3 6.333l.619-.618 4.357 4.357z"\n\t\t\t/>\n\t\t</svg>\n\t'});class Anchor extends FoundationElement{constructor(){super(...arguments),this.handleUnsupportedDelegatesFocus=()=>{var e;window.ShadowRoot&&!window.ShadowRoot.prototype.hasOwnProperty("delegatesFocus")&&(null===(e=this.$fastController.definition.shadowOptions)||void 0===e?void 0:e.delegatesFocus)&&(this.focus=()=>{this.control.focus()})}}connectedCallback(){super.connectedCallback(),this.handleUnsupportedDelegatesFocus()}}(0,tslib_es6.Cg)([attr],Anchor.prototype,"download",void 0),(0,tslib_es6.Cg)([attr],Anchor.prototype,"href",void 0),(0,tslib_es6.Cg)([attr],Anchor.prototype,"hreflang",void 0),(0,tslib_es6.Cg)([attr],Anchor.prototype,"ping",void 0),(0,tslib_es6.Cg)([attr],Anchor.prototype,"referrerpolicy",void 0),(0,tslib_es6.Cg)([attr],Anchor.prototype,"rel",void 0),(0,tslib_es6.Cg)([attr],Anchor.prototype,"target",void 0),(0,tslib_es6.Cg)([attr],Anchor.prototype,"type",void 0),(0,tslib_es6.Cg)([observable],Anchor.prototype,"defaultSlottedContent",void 0);class DelegatesARIALink{}(0,tslib_es6.Cg)([attr({attribute:"aria-expanded"})],DelegatesARIALink.prototype,"ariaExpanded",void 0),applyMixins(DelegatesARIALink,ARIAGlobalStatesAndProperties),applyMixins(Anchor,StartEnd,DelegatesARIALink);const anchorTemplate=(e,t)=>html`
    <a
        class="control"
        part="control"
        download="${e=>e.download}"
        href="${e=>e.href}"
        hreflang="${e=>e.hreflang}"
        ping="${e=>e.ping}"
        referrerpolicy="${e=>e.referrerpolicy}"
        rel="${e=>e.rel}"
        target="${e=>e.target}"
        type="${e=>e.type}"
        aria-atomic="${e=>e.ariaAtomic}"
        aria-busy="${e=>e.ariaBusy}"
        aria-controls="${e=>e.ariaControls}"
        aria-current="${e=>e.ariaCurrent}"
        aria-describedby="${e=>e.ariaDescribedby}"
        aria-details="${e=>e.ariaDetails}"
        aria-disabled="${e=>e.ariaDisabled}"
        aria-errormessage="${e=>e.ariaErrormessage}"
        aria-expanded="${e=>e.ariaExpanded}"
        aria-flowto="${e=>e.ariaFlowto}"
        aria-haspopup="${e=>e.ariaHaspopup}"
        aria-hidden="${e=>e.ariaHidden}"
        aria-invalid="${e=>e.ariaInvalid}"
        aria-keyshortcuts="${e=>e.ariaKeyshortcuts}"
        aria-label="${e=>e.ariaLabel}"
        aria-labelledby="${e=>e.ariaLabelledby}"
        aria-live="${e=>e.ariaLive}"
        aria-owns="${e=>e.ariaOwns}"
        aria-relevant="${e=>e.ariaRelevant}"
        aria-roledescription="${e=>e.ariaRoledescription}"
        ${ref("control")}
    >
        ${startSlotTemplate(e,t)}
        <span class="content" part="content">
            <slot ${slotted("defaultSlottedContent")}></slot>
        </span>
        ${endSlotTemplate(e,t)}
    </a>
`,linkStyles=(e,t)=>css_css`
	${display("inline-flex")} :host {
		background: transparent;
		box-sizing: border-box;
		color: ${linkForeground};
		cursor: pointer;
		fill: currentcolor;
		font-family: ${fontFamily};
		font-size: ${typeRampBaseFontSize};
		line-height: ${typeRampBaseLineHeight};
		outline: none;
	}
	.control {
		background: transparent;
		border: calc(${borderWidth} * 1px) solid transparent;
		border-radius: calc(${cornerRadius} * 1px);
		box-sizing: border-box;
		color: inherit;
		cursor: inherit;
		fill: inherit;
		font-family: inherit;
		height: inherit;
		padding: 0;
		outline: none;
		text-decoration: none;
		word-break: break-word;
	}
	.control::-moz-focus-inner {
		border: 0;
	}
	:host(:hover) {
		color: ${linkActiveForeground};
	}
	:host(:hover) .content {
		text-decoration: underline;
	}
	:host(:active) {
		background: transparent;
		color: ${linkActiveForeground};
	}
	:host(:${focusVisible}) .control,
	:host(:focus) .control {
		border: calc(${borderWidth} * 1px) solid ${focusBorder};
	}
`;class Link extends Anchor{}const vsCodeLink=Link.compose({baseName:"link",template:anchorTemplate,styles:linkStyles,shadowOptions:{delegatesFocus:!0}}),listboxOptionTemplate=(e,t)=>html`
    <template
        aria-checked="${e=>e.ariaChecked}"
        aria-disabled="${e=>e.ariaDisabled}"
        aria-posinset="${e=>e.ariaPosInSet}"
        aria-selected="${e=>e.ariaSelected}"
        aria-setsize="${e=>e.ariaSetSize}"
        class="${e=>[e.checked&&"checked",e.selected&&"selected",e.disabled&&"disabled"].filter(Boolean).join(" ")}"
        role="option"
    >
        ${startSlotTemplate(e,t)}
        <span class="content" part="content">
            <slot ${slotted("content")}></slot>
        </span>
        ${endSlotTemplate(e,t)}
    </template>
`,optionStyles=(e,t)=>css_css`
	${display("inline-flex")} :host {
		font-family: var(--body-font);
		border-radius: ${cornerRadius};
		border: calc(${borderWidth} * 1px) solid transparent;
		box-sizing: border-box;
		color: ${foreground};
		cursor: pointer;
		fill: currentcolor;
		font-size: ${typeRampBaseFontSize};
		line-height: ${typeRampBaseLineHeight};
		margin: 0;
		outline: none;
		overflow: hidden;
		padding: 0 calc((${designUnit} / 2) * 1px)
			calc((${designUnit} / 4) * 1px);
		user-select: none;
		white-space: nowrap;
	}
	:host(:${focusVisible}) {
		border-color: ${focusBorder};
		background: ${listActiveSelectionBackground};
		color: ${foreground};
	}
	:host([aria-selected='true']) {
		background: ${listActiveSelectionBackground};
		border: calc(${borderWidth} * 1px) solid ${focusBorder};
		color: ${listActiveSelectionForeground};
	}
	:host(:active) {
		background: ${listActiveSelectionBackground};
		color: ${listActiveSelectionForeground};
	}
	:host(:not([aria-selected='true']):hover) {
		background: ${listActiveSelectionBackground};
		border: calc(${borderWidth} * 1px) solid ${focusBorder};
		color: ${listActiveSelectionForeground};
	}
	:host(:not([aria-selected='true']):active) {
		background: ${listActiveSelectionBackground};
		color: ${foreground};
	}
	:host([disabled]) {
		cursor: ${disabledCursor};
		opacity: ${disabledOpacity};
	}
	:host([disabled]:hover) {
		background-color: inherit;
	}
	.content {
		grid-column-start: 2;
		justify-self: start;
		overflow: hidden;
		text-overflow: ellipsis;
	}
`;class option_Option extends ListboxOption{connectedCallback(){super.connectedCallback(),this.textContent?this.setAttribute("aria-label",this.textContent):this.setAttribute("aria-label","Option")}}const vsCodeOption=option_Option.compose({baseName:"option",template:listboxOptionTemplate,styles:optionStyles}),TabsOrientation={vertical:"vertical",horizontal:"horizontal"};class Tabs extends FoundationElement{constructor(){super(...arguments),this.orientation=TabsOrientation.horizontal,this.activeindicator=!0,this.showActiveIndicator=!0,this.prevActiveTabIndex=0,this.activeTabIndex=0,this.ticking=!1,this.change=()=>{this.$emit("change",this.activetab)},this.isDisabledElement=e=>"true"===e.getAttribute("aria-disabled"),this.isFocusableElement=e=>!this.isDisabledElement(e),this.setTabs=()=>{const e="gridColumn",t="gridRow",n=this.isHorizontal()?e:t;this.activeTabIndex=this.getActiveIndex(),this.showActiveIndicator=!1,this.tabs.forEach(((r,o)=>{if("tab"===r.slot){const e=this.activeTabIndex===o&&this.isFocusableElement(r);this.activeindicator&&this.isFocusableElement(r)&&(this.showActiveIndicator=!0);const t=this.tabIds[o],n=this.tabpanelIds[o];r.setAttribute("id",t),r.setAttribute("aria-selected",e?"true":"false"),r.setAttribute("aria-controls",n),r.addEventListener("click",this.handleTabClick),r.addEventListener("keydown",this.handleTabKeyDown),r.setAttribute("tabindex",e?"0":"-1"),e&&(this.activetab=r)}r.style[e]="",r.style[t]="",r.style[n]=`${o+1}`,this.isHorizontal()?r.classList.remove("vertical"):r.classList.add("vertical")}))},this.setTabPanels=()=>{this.tabpanels.forEach(((e,t)=>{const n=this.tabIds[t],r=this.tabpanelIds[t];e.setAttribute("id",r),e.setAttribute("aria-labelledby",n),this.activeTabIndex!==t?e.setAttribute("hidden",""):e.removeAttribute("hidden")}))},this.handleTabClick=e=>{const t=e.currentTarget;1===t.nodeType&&this.isFocusableElement(t)&&(this.prevActiveTabIndex=this.activeTabIndex,this.activeTabIndex=this.tabs.indexOf(t),this.setComponent())},this.handleTabKeyDown=e=>{if(this.isHorizontal())switch(e.key){case keyArrowLeft:e.preventDefault(),this.adjustBackward(e);break;case keyArrowRight:e.preventDefault(),this.adjustForward(e)}else switch(e.key){case keyArrowUp:e.preventDefault(),this.adjustBackward(e);break;case keyArrowDown:e.preventDefault(),this.adjustForward(e)}switch(e.key){case keyHome:e.preventDefault(),this.adjust(-this.activeTabIndex);break;case keyEnd:e.preventDefault(),this.adjust(this.tabs.length-this.activeTabIndex-1)}},this.adjustForward=e=>{const t=this.tabs;let n=0;for(n=this.activetab?t.indexOf(this.activetab)+1:1,n===t.length&&(n=0);n<t.length&&t.length>1;){if(this.isFocusableElement(t[n])){this.moveToTabByIndex(t,n);break}if(this.activetab&&n===t.indexOf(this.activetab))break;n+1>=t.length?n=0:n+=1}},this.adjustBackward=e=>{const t=this.tabs;let n=0;for(n=this.activetab?t.indexOf(this.activetab)-1:0,n=n<0?t.length-1:n;n>=0&&t.length>1;){if(this.isFocusableElement(t[n])){this.moveToTabByIndex(t,n);break}n-1<0?n=t.length-1:n-=1}},this.moveToTabByIndex=(e,t)=>{const n=e[t];this.activetab=n,this.prevActiveTabIndex=this.activeTabIndex,this.activeTabIndex=t,n.focus(),this.setComponent()}}orientationChanged(){this.$fastController.isConnected&&(this.setTabs(),this.setTabPanels(),this.handleActiveIndicatorPosition())}activeidChanged(e,t){this.$fastController.isConnected&&this.tabs.length<=this.tabpanels.length&&(this.prevActiveTabIndex=this.tabs.findIndex((t=>t.id===e)),this.setTabs(),this.setTabPanels(),this.handleActiveIndicatorPosition())}tabsChanged(){this.$fastController.isConnected&&this.tabs.length<=this.tabpanels.length&&(this.tabIds=this.getTabIds(),this.tabpanelIds=this.getTabPanelIds(),this.setTabs(),this.setTabPanels(),this.handleActiveIndicatorPosition())}tabpanelsChanged(){this.$fastController.isConnected&&this.tabpanels.length<=this.tabs.length&&(this.tabIds=this.getTabIds(),this.tabpanelIds=this.getTabPanelIds(),this.setTabs(),this.setTabPanels(),this.handleActiveIndicatorPosition())}getActiveIndex(){return void 0!==this.activeid?-1===this.tabIds.indexOf(this.activeid)?0:this.tabIds.indexOf(this.activeid):0}getTabIds(){return this.tabs.map((e=>{var t;return null!==(t=e.getAttribute("id"))&&void 0!==t?t:`tab-${uniqueId()}`}))}getTabPanelIds(){return this.tabpanels.map((e=>{var t;return null!==(t=e.getAttribute("id"))&&void 0!==t?t:`panel-${uniqueId()}`}))}setComponent(){this.activeTabIndex!==this.prevActiveTabIndex&&(this.activeid=this.tabIds[this.activeTabIndex],this.focusTab(),this.change())}isHorizontal(){return this.orientation===TabsOrientation.horizontal}handleActiveIndicatorPosition(){this.showActiveIndicator&&this.activeindicator&&this.activeTabIndex!==this.prevActiveTabIndex&&(this.ticking?this.ticking=!1:(this.ticking=!0,this.animateActiveIndicator()))}animateActiveIndicator(){this.ticking=!0;const e=this.isHorizontal()?"gridColumn":"gridRow",t=this.isHorizontal()?"translateX":"translateY",n=this.isHorizontal()?"offsetLeft":"offsetTop",r=this.activeIndicatorRef[n];this.activeIndicatorRef.style[e]=`${this.activeTabIndex+1}`;const o=this.activeIndicatorRef[n];this.activeIndicatorRef.style[e]=`${this.prevActiveTabIndex+1}`;const i=o-r;this.activeIndicatorRef.style.transform=`${t}(${i}px)`,this.activeIndicatorRef.classList.add("activeIndicatorTransition"),this.activeIndicatorRef.addEventListener("transitionend",(()=>{this.ticking=!1,this.activeIndicatorRef.style[e]=`${this.activeTabIndex+1}`,this.activeIndicatorRef.style.transform=`${t}(0px)`,this.activeIndicatorRef.classList.remove("activeIndicatorTransition")}))}adjust(e){this.prevActiveTabIndex=this.activeTabIndex,this.activeTabIndex=wrapInBounds(0,this.tabs.length-1,this.activeTabIndex+e),this.setComponent()}focusTab(){this.tabs[this.activeTabIndex].focus()}connectedCallback(){super.connectedCallback(),this.tabIds=this.getTabIds(),this.tabpanelIds=this.getTabPanelIds(),this.activeTabIndex=this.getActiveIndex()}}(0,tslib_es6.Cg)([attr],Tabs.prototype,"orientation",void 0),(0,tslib_es6.Cg)([attr],Tabs.prototype,"activeid",void 0),(0,tslib_es6.Cg)([observable],Tabs.prototype,"tabs",void 0),(0,tslib_es6.Cg)([observable],Tabs.prototype,"tabpanels",void 0),(0,tslib_es6.Cg)([attr({mode:"boolean"})],Tabs.prototype,"activeindicator",void 0),(0,tslib_es6.Cg)([observable],Tabs.prototype,"activeIndicatorRef",void 0),(0,tslib_es6.Cg)([observable],Tabs.prototype,"showActiveIndicator",void 0),applyMixins(Tabs,StartEnd);const tabsTemplate=(e,t)=>html`
    <template class="${e=>e.orientation}">
        ${startSlotTemplate(e,t)}
        <div class="tablist" part="tablist" role="tablist">
            <slot class="tab" name="tab" part="tab" ${slotted("tabs")}></slot>

            ${when((e=>e.showActiveIndicator),html`
                    <div
                        ${ref("activeIndicatorRef")}
                        class="activeIndicator"
                        part="activeIndicator"
                    ></div>
                `)}
        </div>
        ${endSlotTemplate(e,t)}
        <div class="tabpanel">
            <slot name="tabpanel" part="tabpanel" ${slotted("tabpanels")}></slot>
        </div>
    </template>
`;class Tab extends FoundationElement{}(0,tslib_es6.Cg)([attr({mode:"boolean"})],Tab.prototype,"disabled",void 0);const tabTemplate=(e,t)=>html`
    <template slot="tab" role="tab" aria-disabled="${e=>e.disabled}">
        <slot></slot>
    </template>
`;class TabPanel extends FoundationElement{}const tabPanelTemplate=(e,t)=>html`
    <template slot="tabpanel" role="tabpanel">
        <slot></slot>
    </template>
`,panelsStyles=(e,t)=>css_css`
	${display("grid")} :host {
		box-sizing: border-box;
		font-family: ${fontFamily};
		font-size: ${typeRampBaseFontSize};
		line-height: ${typeRampBaseLineHeight};
		color: ${foreground};
		grid-template-columns: auto 1fr auto;
		grid-template-rows: auto 1fr;
		overflow-x: auto;
	}
	.tablist {
		display: grid;
		grid-template-rows: auto auto;
		grid-template-columns: auto;
		column-gap: calc(${designUnit} * 8px);
		position: relative;
		width: max-content;
		align-self: end;
		padding: calc(${designUnit} * 1px) calc(${designUnit} * 1px) 0;
		box-sizing: border-box;
	}
	.start,
	.end {
		align-self: center;
	}
	.activeIndicator {
		grid-row: 2;
		grid-column: 1;
		width: 100%;
		height: calc((${designUnit} / 4) * 1px);
		justify-self: center;
		background: ${panelTabActiveForeground};
		margin: 0;
		border-radius: calc(${cornerRadius} * 1px);
	}
	.activeIndicatorTransition {
		transition: transform 0.01s linear;
	}
	.tabpanel {
		grid-row: 2;
		grid-column-start: 1;
		grid-column-end: 4;
		position: relative;
	}
`,panelTabStyles=(e,t)=>css_css`
	${display("inline-flex")} :host {
		box-sizing: border-box;
		font-family: ${fontFamily};
		font-size: ${typeRampBaseFontSize};
		line-height: ${typeRampBaseLineHeight};
		height: calc(${designUnit} * 7px);
		padding: calc(${designUnit} * 1px) 0;
		color: ${panelTabForeground};
		fill: currentcolor;
		border-radius: calc(${cornerRadius} * 1px);
		border: solid calc(${borderWidth} * 1px) transparent;
		align-items: center;
		justify-content: center;
		grid-row: 1;
		cursor: pointer;
	}
	:host(:hover) {
		color: ${panelTabActiveForeground};
		fill: currentcolor;
	}
	:host(:active) {
		color: ${panelTabActiveForeground};
		fill: currentcolor;
	}
	:host([aria-selected='true']) {
		background: transparent;
		color: ${panelTabActiveForeground};
		fill: currentcolor;
	}
	:host([aria-selected='true']:hover) {
		background: transparent;
		color: ${panelTabActiveForeground};
		fill: currentcolor;
	}
	:host([aria-selected='true']:active) {
		background: transparent;
		color: ${panelTabActiveForeground};
		fill: currentcolor;
	}
	:host(:${focusVisible}) {
		outline: none;
		border: solid calc(${borderWidth} * 1px) ${panelTabActiveBorder};
	}
	:host(:focus) {
		outline: none;
	}
	::slotted(vscode-badge) {
		margin-inline-start: calc(${designUnit} * 2px);
	}
`,panelViewStyles=(e,t)=>css_css`
	${display("flex")} :host {
		color: inherit;
		background-color: transparent;
		border: solid calc(${borderWidth} * 1px) transparent;
		box-sizing: border-box;
		font-size: ${typeRampBaseFontSize};
		line-height: ${typeRampBaseLineHeight};
		padding: 10px calc((${designUnit} + 2) * 1px);
	}
`;class Panels extends Tabs{connectedCallback(){super.connectedCallback(),this.orientation&&(this.orientation=TabsOrientation.horizontal),this.getAttribute("aria-label")||this.setAttribute("aria-label","Panels")}}const vsCodePanels=Panels.compose({baseName:"panels",template:tabsTemplate,styles:panelsStyles});class PanelTab extends Tab{connectedCallback(){super.connectedCallback(),this.disabled&&(this.disabled=!1),this.textContent&&this.setAttribute("aria-label",this.textContent)}}const vsCodePanelTab=PanelTab.compose({baseName:"panel-tab",template:tabTemplate,styles:panelTabStyles});class PanelView extends TabPanel{}const vsCodePanelView=PanelView.compose({baseName:"panel-view",template:tabPanelTemplate,styles:panelViewStyles});class BaseProgress extends FoundationElement{constructor(){super(...arguments),this.percentComplete=0}valueChanged(){this.$fastController.isConnected&&this.updatePercentComplete()}minChanged(){this.$fastController.isConnected&&this.updatePercentComplete()}maxChanged(){this.$fastController.isConnected&&this.updatePercentComplete()}connectedCallback(){super.connectedCallback(),this.updatePercentComplete()}updatePercentComplete(){const e="number"==typeof this.min?this.min:0,t="number"==typeof this.max?this.max:100,n="number"==typeof this.value?this.value:0,r=t-e;this.percentComplete=0===r?0:Math.fround((n-e)/r*100)}}(0,tslib_es6.Cg)([attr({converter:nullableNumberConverter})],BaseProgress.prototype,"value",void 0),(0,tslib_es6.Cg)([attr({converter:nullableNumberConverter})],BaseProgress.prototype,"min",void 0),(0,tslib_es6.Cg)([attr({converter:nullableNumberConverter})],BaseProgress.prototype,"max",void 0),(0,tslib_es6.Cg)([attr({mode:"boolean"})],BaseProgress.prototype,"paused",void 0),(0,tslib_es6.Cg)([observable],BaseProgress.prototype,"percentComplete",void 0);const progressSegments=44,progressRingTemplate=(e,t)=>html`
    <template
        role="progressbar"
        aria-valuenow="${e=>e.value}"
        aria-valuemin="${e=>e.min}"
        aria-valuemax="${e=>e.max}"
        class="${e=>e.paused?"paused":""}"
    >
        ${when((e=>"number"==typeof e.value),html`
                <svg
                    class="progress"
                    part="progress"
                    viewBox="0 0 16 16"
                    slot="determinate"
                >
                    <circle
                        class="background"
                        part="background"
                        cx="8px"
                        cy="8px"
                        r="7px"
                    ></circle>
                    <circle
                        class="determinate"
                        part="determinate"
                        style="stroke-dasharray: ${e=>progressSegments*e.percentComplete/100}px ${progressSegments}px"
                        cx="8px"
                        cy="8px"
                        r="7px"
                    ></circle>
                </svg>
            `)}
        ${when((e=>"number"!=typeof e.value),html`
                <slot name="indeterminate" slot="indeterminate">
                    ${t.indeterminateIndicator||""}
                </slot>
            `)}
    </template>
`,progressRingStyles=(e,t)=>css_css`
	${display("flex")} :host {
		align-items: center;
		outline: none;
		height: calc(${designUnit} * 7px);
		width: calc(${designUnit} * 7px);
		margin: 0;
	}
	.progress {
		height: 100%;
		width: 100%;
	}
	.background {
		fill: none;
		stroke: transparent;
		stroke-width: calc(${designUnit} / 2 * 1px);
	}
	.indeterminate-indicator-1 {
		fill: none;
		stroke: ${progressBackground};
		stroke-width: calc(${designUnit} / 2 * 1px);
		stroke-linecap: square;
		transform-origin: 50% 50%;
		transform: rotate(-90deg);
		transition: all 0.2s ease-in-out;
		animation: spin-infinite 2s linear infinite;
	}
	@keyframes spin-infinite {
		0% {
			stroke-dasharray: 0.01px 43.97px;
			transform: rotate(0deg);
		}
		50% {
			stroke-dasharray: 21.99px 21.99px;
			transform: rotate(450deg);
		}
		100% {
			stroke-dasharray: 0.01px 43.97px;
			transform: rotate(1080deg);
		}
	}
`;class ProgressRing extends BaseProgress{connectedCallback(){super.connectedCallback(),this.paused&&(this.paused=!1),this.setAttribute("aria-label","Loading"),this.setAttribute("aria-live","assertive"),this.setAttribute("role","alert")}attributeChangedCallback(e,t,n){"value"===e&&this.removeAttribute("value")}}const vsCodeProgressRing=ProgressRing.compose({baseName:"progress-ring",template:progressRingTemplate,styles:progressRingStyles,indeterminateIndicator:'\n\t\t<svg class="progress" part="progress" viewBox="0 0 16 16">\n\t\t\t<circle\n\t\t\t\tclass="background"\n\t\t\t\tpart="background"\n\t\t\t\tcx="8px"\n\t\t\t\tcy="8px"\n\t\t\t\tr="7px"\n\t\t\t></circle>\n\t\t\t<circle\n\t\t\t\tclass="indeterminate-indicator-1"\n\t\t\t\tpart="indeterminate-indicator-1"\n\t\t\t\tcx="8px"\n\t\t\t\tcy="8px"\n\t\t\t\tr="7px"\n\t\t\t></circle>\n\t\t</svg>\n\t'});class _Radio extends FoundationElement{}class FormAssociatedRadio extends(CheckableFormAssociated(_Radio)){constructor(){super(...arguments),this.proxy=document.createElement("input")}}class radio_Radio extends FormAssociatedRadio{constructor(){super(),this.initialValue="on",this.keypressHandler=e=>{if(e.key!==keySpace)return!0;this.checked||this.readOnly||(this.checked=!0)},this.proxy.setAttribute("type","radio")}readOnlyChanged(){this.proxy instanceof HTMLInputElement&&(this.proxy.readOnly=this.readOnly)}defaultCheckedChanged(){var e;this.$fastController.isConnected&&!this.dirtyChecked&&(this.isInsideRadioGroup()||(this.checked=null!==(e=this.defaultChecked)&&void 0!==e&&e,this.dirtyChecked=!1))}connectedCallback(){var e,t;super.connectedCallback(),this.validate(),"radiogroup"!==(null===(e=this.parentElement)||void 0===e?void 0:e.getAttribute("role"))&&null===this.getAttribute("tabindex")&&(this.disabled||this.setAttribute("tabindex","0")),this.checkedAttribute&&(this.dirtyChecked||this.isInsideRadioGroup()||(this.checked=null!==(t=this.defaultChecked)&&void 0!==t&&t,this.dirtyChecked=!1))}isInsideRadioGroup(){return null!==this.closest("[role=radiogroup]")}clickHandler(e){this.disabled||this.readOnly||this.checked||(this.checked=!0)}}(0,tslib_es6.Cg)([attr({attribute:"readonly",mode:"boolean"})],radio_Radio.prototype,"readOnly",void 0),(0,tslib_es6.Cg)([observable],radio_Radio.prototype,"name",void 0),(0,tslib_es6.Cg)([observable],radio_Radio.prototype,"defaultSlottedNodes",void 0);const radioTemplate=(e,t)=>html`
    <template
        role="radio"
        class="${e=>e.checked?"checked":""} ${e=>e.readOnly?"readonly":""}"
        aria-checked="${e=>e.checked}"
        aria-required="${e=>e.required}"
        aria-disabled="${e=>e.disabled}"
        aria-readonly="${e=>e.readOnly}"
        @keypress="${(e,t)=>e.keypressHandler(t.event)}"
        @click="${(e,t)=>e.clickHandler(t.event)}"
    >
        <div part="control" class="control">
            <slot name="checked-indicator">
                ${t.checkedIndicator||""}
            </slot>
        </div>
        <label
            part="label"
            class="${e=>e.defaultSlottedNodes&&e.defaultSlottedNodes.length?"label":"label label__hidden"}"
        >
            <slot ${slotted("defaultSlottedNodes")}></slot>
        </label>
    </template>
`,radioStyles=(e,t)=>css_css`
	${display("inline-flex")} :host {
		align-items: center;
		flex-direction: row;
		font-size: ${typeRampBaseFontSize};
		line-height: ${typeRampBaseLineHeight};
		margin: calc(${designUnit} * 1px) 0;
		outline: none;
		position: relative;
		transition: all 0.2s ease-in-out;
		user-select: none;
	}
	.control {
		background: ${checkboxBackground};
		border-radius: 999px;
		border: calc(${borderWidth} * 1px) solid ${checkboxBorder};
		box-sizing: border-box;
		cursor: pointer;
		height: calc(${designUnit} * 4px);
		position: relative;
		outline: none;
		width: calc(${designUnit} * 4px);
	}
	.label {
		color: ${foreground};
		cursor: pointer;
		font-family: ${fontFamily};
		margin-inline-end: calc(${designUnit} * 2px + 2px);
		padding-inline-start: calc(${designUnit} * 2px + 2px);
	}
	.label__hidden {
		display: none;
		visibility: hidden;
	}
	.control,
	.checked-indicator {
		flex-shrink: 0;
	}
	.checked-indicator {
		background: ${foreground};
		border-radius: 999px;
		display: inline-block;
		inset: calc(${designUnit} * 1px);
		opacity: 0;
		pointer-events: none;
		position: absolute;
	}
	:host(:not([disabled])) .control:hover {
		background: ${checkboxBackground};
		border-color: ${checkboxBorder};
	}
	:host(:not([disabled])) .control:active {
		background: ${checkboxBackground};
		border-color: ${focusBorder};
	}
	:host(:${focusVisible}) .control {
		border: calc(${borderWidth} * 1px) solid ${focusBorder};
	}
	:host([aria-checked='true']) .control {
		background: ${checkboxBackground};
		border: calc(${borderWidth} * 1px) solid ${checkboxBorder};
	}
	:host([aria-checked='true']:not([disabled])) .control:hover {
		background: ${checkboxBackground};
		border: calc(${borderWidth} * 1px) solid ${checkboxBorder};
	}
	:host([aria-checked='true']:not([disabled])) .control:active {
		background: ${checkboxBackground};
		border: calc(${borderWidth} * 1px) solid ${focusBorder};
	}
	:host([aria-checked="true"]:${focusVisible}:not([disabled])) .control {
		border: calc(${borderWidth} * 1px) solid ${focusBorder};
	}
	:host([disabled]) .label,
	:host([readonly]) .label,
	:host([readonly]) .control,
	:host([disabled]) .control {
		cursor: ${disabledCursor};
	}
	:host([aria-checked='true']) .checked-indicator {
		opacity: 1;
	}
	:host([disabled]) {
		opacity: ${disabledOpacity};
	}
`;class Radio extends radio_Radio{connectedCallback(){super.connectedCallback(),this.textContent?this.setAttribute("aria-label",this.textContent):this.setAttribute("aria-label","Radio")}}const vsCodeRadio=Radio.compose({baseName:"radio",template:radioTemplate,styles:radioStyles,checkedIndicator:'\n\t\t<div part="checked-indicator" class="checked-indicator"></div>\n\t'});var localization_Direction;!function(e){e.ltr="ltr",e.rtl="rtl"}(localization_Direction||(localization_Direction={}));const getDirection=e=>{const t=e.closest("[dir]");return null!==t&&"rtl"===t.dir?localization_Direction.rtl:localization_Direction.ltr};class radio_group_RadioGroup extends FoundationElement{constructor(){super(...arguments),this.orientation=Orientation.horizontal,this.radioChangeHandler=e=>{const t=e.target;t.checked&&(this.slottedRadioButtons.forEach((e=>{e!==t&&(e.checked=!1,this.isInsideFoundationToolbar||e.setAttribute("tabindex","-1"))})),this.selectedRadio=t,this.value=t.value,t.setAttribute("tabindex","0"),this.focusedRadio=t),e.stopPropagation()},this.moveToRadioByIndex=(e,t)=>{const n=e[t];this.isInsideToolbar||(n.setAttribute("tabindex","0"),n.readOnly?this.slottedRadioButtons.forEach((e=>{e!==n&&e.setAttribute("tabindex","-1")})):(n.checked=!0,this.selectedRadio=n)),this.focusedRadio=n,n.focus()},this.moveRightOffGroup=()=>{var e;null===(e=this.nextElementSibling)||void 0===e||e.focus()},this.moveLeftOffGroup=()=>{var e;null===(e=this.previousElementSibling)||void 0===e||e.focus()},this.focusOutHandler=e=>{const t=this.slottedRadioButtons,n=e.target,r=null!==n?t.indexOf(n):0,o=this.focusedRadio?t.indexOf(this.focusedRadio):-1;return(0===o&&r===o||o===t.length-1&&o===r)&&(this.selectedRadio?(this.focusedRadio=this.selectedRadio,this.isInsideFoundationToolbar||(this.selectedRadio.setAttribute("tabindex","0"),t.forEach((e=>{e!==this.selectedRadio&&e.setAttribute("tabindex","-1")})))):(this.focusedRadio=t[0],this.focusedRadio.setAttribute("tabindex","0"),t.forEach((e=>{e!==this.focusedRadio&&e.setAttribute("tabindex","-1")})))),!0},this.clickHandler=e=>{const t=e.target;if(t){const e=this.slottedRadioButtons;t.checked||0===e.indexOf(t)?(t.setAttribute("tabindex","0"),this.selectedRadio=t):(t.setAttribute("tabindex","-1"),this.selectedRadio=null),this.focusedRadio=t}e.preventDefault()},this.shouldMoveOffGroupToTheRight=(e,t,n)=>e===t.length&&this.isInsideToolbar&&n===keyArrowRight,this.shouldMoveOffGroupToTheLeft=(e,t)=>(this.focusedRadio?e.indexOf(this.focusedRadio)-1:0)<0&&this.isInsideToolbar&&t===keyArrowLeft,this.checkFocusedRadio=()=>{null===this.focusedRadio||this.focusedRadio.readOnly||this.focusedRadio.checked||(this.focusedRadio.checked=!0,this.focusedRadio.setAttribute("tabindex","0"),this.focusedRadio.focus(),this.selectedRadio=this.focusedRadio)},this.moveRight=e=>{const t=this.slottedRadioButtons;let n=0;if(n=this.focusedRadio?t.indexOf(this.focusedRadio)+1:1,this.shouldMoveOffGroupToTheRight(n,t,e.key))this.moveRightOffGroup();else for(n===t.length&&(n=0);n<t.length&&t.length>1;){if(!t[n].disabled){this.moveToRadioByIndex(t,n);break}if(this.focusedRadio&&n===t.indexOf(this.focusedRadio))break;if(n+1>=t.length){if(this.isInsideToolbar)break;n=0}else n+=1}},this.moveLeft=e=>{const t=this.slottedRadioButtons;let n=0;if(n=this.focusedRadio?t.indexOf(this.focusedRadio)-1:0,n=n<0?t.length-1:n,this.shouldMoveOffGroupToTheLeft(t,e.key))this.moveLeftOffGroup();else for(;n>=0&&t.length>1;){if(!t[n].disabled){this.moveToRadioByIndex(t,n);break}if(this.focusedRadio&&n===t.indexOf(this.focusedRadio))break;n-1<0?n=t.length-1:n-=1}},this.keydownHandler=e=>{const t=e.key;if(t in ArrowKeys&&this.isInsideFoundationToolbar)return!0;switch(t){case keyEnter:this.checkFocusedRadio();break;case keyArrowRight:case keyArrowDown:this.direction===localization_Direction.ltr?this.moveRight(e):this.moveLeft(e);break;case keyArrowLeft:case keyArrowUp:this.direction===localization_Direction.ltr?this.moveLeft(e):this.moveRight(e);break;default:return!0}}}readOnlyChanged(){void 0!==this.slottedRadioButtons&&this.slottedRadioButtons.forEach((e=>{this.readOnly?e.readOnly=!0:e.readOnly=!1}))}disabledChanged(){void 0!==this.slottedRadioButtons&&this.slottedRadioButtons.forEach((e=>{this.disabled?e.disabled=!0:e.disabled=!1}))}nameChanged(){this.slottedRadioButtons&&this.slottedRadioButtons.forEach((e=>{e.setAttribute("name",this.name)}))}valueChanged(){this.slottedRadioButtons&&this.slottedRadioButtons.forEach((e=>{e.value===this.value&&(e.checked=!0,this.selectedRadio=e)})),this.$emit("change")}slottedRadioButtonsChanged(e,t){this.slottedRadioButtons&&this.slottedRadioButtons.length>0&&this.setupRadioButtons()}get parentToolbar(){return this.closest('[role="toolbar"]')}get isInsideToolbar(){var e;return null!==(e=this.parentToolbar)&&void 0!==e&&e}get isInsideFoundationToolbar(){var e;return!!(null===(e=this.parentToolbar)||void 0===e?void 0:e.$fastController)}connectedCallback(){super.connectedCallback(),this.direction=getDirection(this),this.setupRadioButtons()}disconnectedCallback(){this.slottedRadioButtons.forEach((e=>{e.removeEventListener("change",this.radioChangeHandler)}))}setupRadioButtons(){const e=this.slottedRadioButtons.filter((e=>e.hasAttribute("checked"))),t=e?e.length:0;t>1&&(e[t-1].checked=!0);let n=!1;if(this.slottedRadioButtons.forEach((e=>{void 0!==this.name&&e.setAttribute("name",this.name),this.disabled&&(e.disabled=!0),this.readOnly&&(e.readOnly=!0),this.value&&this.value===e.value?(this.selectedRadio=e,this.focusedRadio=e,e.checked=!0,e.setAttribute("tabindex","0"),n=!0):(this.isInsideFoundationToolbar||e.setAttribute("tabindex","-1"),e.checked=!1),e.addEventListener("change",this.radioChangeHandler)})),void 0===this.value&&this.slottedRadioButtons.length>0){const e=this.slottedRadioButtons.filter((e=>e.hasAttribute("checked"))),t=null!==e?e.length:0;if(t>0&&!n){const n=e[t-1];n.checked=!0,this.focusedRadio=n,n.setAttribute("tabindex","0")}else this.slottedRadioButtons[0].setAttribute("tabindex","0"),this.focusedRadio=this.slottedRadioButtons[0]}}}(0,tslib_es6.Cg)([attr({attribute:"readonly",mode:"boolean"})],radio_group_RadioGroup.prototype,"readOnly",void 0),(0,tslib_es6.Cg)([attr({attribute:"disabled",mode:"boolean"})],radio_group_RadioGroup.prototype,"disabled",void 0),(0,tslib_es6.Cg)([attr],radio_group_RadioGroup.prototype,"name",void 0),(0,tslib_es6.Cg)([attr],radio_group_RadioGroup.prototype,"value",void 0),(0,tslib_es6.Cg)([attr],radio_group_RadioGroup.prototype,"orientation",void 0),(0,tslib_es6.Cg)([observable],radio_group_RadioGroup.prototype,"childItems",void 0),(0,tslib_es6.Cg)([observable],radio_group_RadioGroup.prototype,"slottedRadioButtons",void 0);const radioGroupTemplate=(e,t)=>html`
    <template
        role="radiogroup"
        aria-disabled="${e=>e.disabled}"
        aria-readonly="${e=>e.readOnly}"
        @click="${(e,t)=>e.clickHandler(t.event)}"
        @keydown="${(e,t)=>e.keydownHandler(t.event)}"
        @focusout="${(e,t)=>e.focusOutHandler(t.event)}"
    >
        <slot name="label"></slot>
        <div
            class="positioning-region ${e=>e.orientation===Orientation.horizontal?"horizontal":"vertical"}"
            part="positioning-region"
        >
            <slot
                ${slotted({property:"slottedRadioButtons",filter:node_observation_elements("[role=radio]")})}
            ></slot>
        </div>
    </template>
`,radioGroupStyles=(e,t)=>css_css`
	${display("flex")} :host {
		align-items: flex-start;
		margin: calc(${designUnit} * 1px) 0;
		flex-direction: column;
	}
	.positioning-region {
		display: flex;
		flex-wrap: wrap;
	}
	:host([orientation='vertical']) .positioning-region {
		flex-direction: column;
	}
	:host([orientation='horizontal']) .positioning-region {
		flex-direction: row;
	}
	::slotted([slot='label']) {
		color: ${foreground};
		font-size: ${typeRampBaseFontSize};
		margin: calc(${designUnit} * 1px) 0;
	}
`;class RadioGroup extends radio_group_RadioGroup{connectedCallback(){super.connectedCallback();const e=this.querySelector("label");if(e){const t="radio-group-"+Math.random().toString(16).slice(2);e.setAttribute("id",t),this.setAttribute("aria-labelledby",t)}}}const vsCodeRadioGroup=RadioGroup.compose({baseName:"radio-group",template:radioGroupTemplate,styles:radioGroupStyles}),tagStyles=(e,t)=>css_css`
	${display("inline-block")} :host {
		box-sizing: border-box;
		font-family: ${fontFamily};
		font-size: ${typeRampMinus1FontSize};
		line-height: ${typeRampMinus1LineHeight};
	}
	.control {
		background-color: ${badgeBackground};
		border: calc(${borderWidth} * 1px) solid ${buttonBorder};
		border-radius: ${tagCornerRadius};
		color: ${badgeForeground};
		padding: calc(${designUnit} * 0.5px) calc(${designUnit} * 1px);
		text-transform: uppercase;
	}
`;class Tag extends badge_Badge{connectedCallback(){super.connectedCallback(),this.circular&&(this.circular=!1)}}const vsCodeTag=Tag.compose({baseName:"tag",template:badgeTemplate,styles:tagStyles});class _TextField extends FoundationElement{}class FormAssociatedTextField extends(FormAssociated(_TextField)){constructor(){super(...arguments),this.proxy=document.createElement("input")}}const TextFieldType={email:"email",password:"password",tel:"tel",text:"text",url:"url"};class text_field_TextField extends FormAssociatedTextField{constructor(){super(...arguments),this.type=TextFieldType.text}readOnlyChanged(){this.proxy instanceof HTMLInputElement&&(this.proxy.readOnly=this.readOnly,this.validate())}autofocusChanged(){this.proxy instanceof HTMLInputElement&&(this.proxy.autofocus=this.autofocus,this.validate())}placeholderChanged(){this.proxy instanceof HTMLInputElement&&(this.proxy.placeholder=this.placeholder)}typeChanged(){this.proxy instanceof HTMLInputElement&&(this.proxy.type=this.type,this.validate())}listChanged(){this.proxy instanceof HTMLInputElement&&(this.proxy.setAttribute("list",this.list),this.validate())}maxlengthChanged(){this.proxy instanceof HTMLInputElement&&(this.proxy.maxLength=this.maxlength,this.validate())}minlengthChanged(){this.proxy instanceof HTMLInputElement&&(this.proxy.minLength=this.minlength,this.validate())}patternChanged(){this.proxy instanceof HTMLInputElement&&(this.proxy.pattern=this.pattern,this.validate())}sizeChanged(){this.proxy instanceof HTMLInputElement&&(this.proxy.size=this.size)}spellcheckChanged(){this.proxy instanceof HTMLInputElement&&(this.proxy.spellcheck=this.spellcheck)}connectedCallback(){super.connectedCallback(),this.proxy.setAttribute("type",this.type),this.validate(),this.autofocus&&DOM.queueUpdate((()=>{this.focus()}))}select(){this.control.select(),this.$emit("select")}handleTextInput(){this.value=this.control.value}handleChange(){this.$emit("change")}validate(){super.validate(this.control)}}(0,tslib_es6.Cg)([attr({attribute:"readonly",mode:"boolean"})],text_field_TextField.prototype,"readOnly",void 0),(0,tslib_es6.Cg)([attr({mode:"boolean"})],text_field_TextField.prototype,"autofocus",void 0),(0,tslib_es6.Cg)([attr],text_field_TextField.prototype,"placeholder",void 0),(0,tslib_es6.Cg)([attr],text_field_TextField.prototype,"type",void 0),(0,tslib_es6.Cg)([attr],text_field_TextField.prototype,"list",void 0),(0,tslib_es6.Cg)([attr({converter:nullableNumberConverter})],text_field_TextField.prototype,"maxlength",void 0),(0,tslib_es6.Cg)([attr({converter:nullableNumberConverter})],text_field_TextField.prototype,"minlength",void 0),(0,tslib_es6.Cg)([attr],text_field_TextField.prototype,"pattern",void 0),(0,tslib_es6.Cg)([attr({converter:nullableNumberConverter})],text_field_TextField.prototype,"size",void 0),(0,tslib_es6.Cg)([attr({mode:"boolean"})],text_field_TextField.prototype,"spellcheck",void 0),(0,tslib_es6.Cg)([observable],text_field_TextField.prototype,"defaultSlottedNodes",void 0);class DelegatesARIATextbox{}applyMixins(DelegatesARIATextbox,ARIAGlobalStatesAndProperties),applyMixins(text_field_TextField,StartEnd,DelegatesARIATextbox);class _TextArea extends FoundationElement{}class FormAssociatedTextArea extends(FormAssociated(_TextArea)){constructor(){super(...arguments),this.proxy=document.createElement("textarea")}}const TextAreaResize={none:"none",both:"both",horizontal:"horizontal",vertical:"vertical"};class text_area_TextArea extends FormAssociatedTextArea{constructor(){super(...arguments),this.resize=TextAreaResize.none,this.cols=20,this.handleTextInput=()=>{this.value=this.control.value}}readOnlyChanged(){this.proxy instanceof HTMLTextAreaElement&&(this.proxy.readOnly=this.readOnly)}autofocusChanged(){this.proxy instanceof HTMLTextAreaElement&&(this.proxy.autofocus=this.autofocus)}listChanged(){this.proxy instanceof HTMLTextAreaElement&&this.proxy.setAttribute("list",this.list)}maxlengthChanged(){this.proxy instanceof HTMLTextAreaElement&&(this.proxy.maxLength=this.maxlength)}minlengthChanged(){this.proxy instanceof HTMLTextAreaElement&&(this.proxy.minLength=this.minlength)}spellcheckChanged(){this.proxy instanceof HTMLTextAreaElement&&(this.proxy.spellcheck=this.spellcheck)}select(){this.control.select(),this.$emit("select")}handleChange(){this.$emit("change")}validate(){super.validate(this.control)}}(0,tslib_es6.Cg)([attr({mode:"boolean"})],text_area_TextArea.prototype,"readOnly",void 0),(0,tslib_es6.Cg)([attr],text_area_TextArea.prototype,"resize",void 0),(0,tslib_es6.Cg)([attr({mode:"boolean"})],text_area_TextArea.prototype,"autofocus",void 0),(0,tslib_es6.Cg)([attr({attribute:"form"})],text_area_TextArea.prototype,"formId",void 0),(0,tslib_es6.Cg)([attr],text_area_TextArea.prototype,"list",void 0),(0,tslib_es6.Cg)([attr({converter:nullableNumberConverter})],text_area_TextArea.prototype,"maxlength",void 0),(0,tslib_es6.Cg)([attr({converter:nullableNumberConverter})],text_area_TextArea.prototype,"minlength",void 0),(0,tslib_es6.Cg)([attr],text_area_TextArea.prototype,"name",void 0),(0,tslib_es6.Cg)([attr],text_area_TextArea.prototype,"placeholder",void 0),(0,tslib_es6.Cg)([attr({converter:nullableNumberConverter,mode:"fromView"})],text_area_TextArea.prototype,"cols",void 0),(0,tslib_es6.Cg)([attr({converter:nullableNumberConverter,mode:"fromView"})],text_area_TextArea.prototype,"rows",void 0),(0,tslib_es6.Cg)([attr({mode:"boolean"})],text_area_TextArea.prototype,"spellcheck",void 0),(0,tslib_es6.Cg)([observable],text_area_TextArea.prototype,"defaultSlottedNodes",void 0),applyMixins(text_area_TextArea,DelegatesARIATextbox);const textAreaTemplate=(e,t)=>html`
    <template
        class="
            ${e=>e.readOnly?"readonly":""}
            ${e=>e.resize!==TextAreaResize.none?`resize-${e.resize}`:""}"
    >
        <label
            part="label"
            for="control"
            class="${e=>e.defaultSlottedNodes&&e.defaultSlottedNodes.length?"label":"label label__hidden"}"
        >
            <slot ${slotted("defaultSlottedNodes")}></slot>
        </label>
        <textarea
            part="control"
            class="control"
            id="control"
            ?autofocus="${e=>e.autofocus}"
            cols="${e=>e.cols}"
            ?disabled="${e=>e.disabled}"
            form="${e=>e.form}"
            list="${e=>e.list}"
            maxlength="${e=>e.maxlength}"
            minlength="${e=>e.minlength}"
            name="${e=>e.name}"
            placeholder="${e=>e.placeholder}"
            ?readonly="${e=>e.readOnly}"
            ?required="${e=>e.required}"
            rows="${e=>e.rows}"
            ?spellcheck="${e=>e.spellcheck}"
            :value="${e=>e.value}"
            aria-atomic="${e=>e.ariaAtomic}"
            aria-busy="${e=>e.ariaBusy}"
            aria-controls="${e=>e.ariaControls}"
            aria-current="${e=>e.ariaCurrent}"
            aria-describedby="${e=>e.ariaDescribedby}"
            aria-details="${e=>e.ariaDetails}"
            aria-disabled="${e=>e.ariaDisabled}"
            aria-errormessage="${e=>e.ariaErrormessage}"
            aria-flowto="${e=>e.ariaFlowto}"
            aria-haspopup="${e=>e.ariaHaspopup}"
            aria-hidden="${e=>e.ariaHidden}"
            aria-invalid="${e=>e.ariaInvalid}"
            aria-keyshortcuts="${e=>e.ariaKeyshortcuts}"
            aria-label="${e=>e.ariaLabel}"
            aria-labelledby="${e=>e.ariaLabelledby}"
            aria-live="${e=>e.ariaLive}"
            aria-owns="${e=>e.ariaOwns}"
            aria-relevant="${e=>e.ariaRelevant}"
            aria-roledescription="${e=>e.ariaRoledescription}"
            @input="${(e,t)=>e.handleTextInput()}"
            @change="${e=>e.handleChange()}"
            ${ref("control")}
        ></textarea>
    </template>
`,textAreaStyles=(e,t)=>css_css`
	${display("inline-block")} :host {
		font-family: ${fontFamily};
		outline: none;
		user-select: none;
	}
	.control {
		box-sizing: border-box;
		position: relative;
		color: ${inputForeground};
		background: ${inputBackground};
		border-radius: calc(${cornerRadius} * 1px);
		border: calc(${borderWidth} * 1px) solid ${dropdownBorder};
		font: inherit;
		font-size: ${typeRampBaseFontSize};
		line-height: ${typeRampBaseLineHeight};
		padding: calc(${designUnit} * 2px + 1px);
		width: 100%;
		min-width: ${inputMinWidth};
		resize: none;
	}
	.control:hover:enabled {
		background: ${inputBackground};
		border-color: ${dropdownBorder};
	}
	.control:active:enabled {
		background: ${inputBackground};
		border-color: ${focusBorder};
	}
	.control:hover,
	.control:${focusVisible},
	.control:disabled,
	.control:active {
		outline: none;
	}
	.control::-webkit-scrollbar {
		width: ${scrollbarWidth};
		height: ${scrollbarHeight};
	}
	.control::-webkit-scrollbar-corner {
		background: ${inputBackground};
	}
	.control::-webkit-scrollbar-thumb {
		background: ${scrollbarSliderBackground};
	}
	.control::-webkit-scrollbar-thumb:hover {
		background: ${scrollbarSliderHoverBackground};
	}
	.control::-webkit-scrollbar-thumb:active {
		background: ${scrollbarSliderActiveBackground};
	}
	:host(:focus-within:not([disabled])) .control {
		border-color: ${focusBorder};
	}
	:host([resize='both']) .control {
		resize: both;
	}
	:host([resize='horizontal']) .control {
		resize: horizontal;
	}
	:host([resize='vertical']) .control {
		resize: vertical;
	}
	.label {
		display: block;
		color: ${foreground};
		cursor: pointer;
		font-size: ${typeRampBaseFontSize};
		line-height: ${typeRampBaseLineHeight};
		margin-bottom: 2px;
	}
	.label__hidden {
		display: none;
		visibility: hidden;
	}
	:host([disabled]) .label,
	:host([readonly]) .label,
	:host([readonly]) .control,
	:host([disabled]) .control {
		cursor: ${disabledCursor};
	}
	:host([disabled]) {
		opacity: ${disabledOpacity};
	}
	:host([disabled]) .control {
		border-color: ${dropdownBorder};
	}
`;class TextArea extends text_area_TextArea{connectedCallback(){super.connectedCallback(),this.textContent?this.setAttribute("aria-label",this.textContent):this.setAttribute("aria-label","Text area")}}const vsCodeTextArea=TextArea.compose({baseName:"text-area",template:textAreaTemplate,styles:textAreaStyles,shadowOptions:{delegatesFocus:!0}});function whitespaceFilter(e,t,n){return e.nodeType!==Node.TEXT_NODE||"string"==typeof e.nodeValue&&!!e.nodeValue.trim().length}const textFieldTemplate=(e,t)=>html`
    <template
        class="
            ${e=>e.readOnly?"readonly":""}
        "
    >
        <label
            part="label"
            for="control"
            class="${e=>e.defaultSlottedNodes&&e.defaultSlottedNodes.length?"label":"label label__hidden"}"
        >
            <slot
                ${slotted({property:"defaultSlottedNodes",filter:whitespaceFilter})}
            ></slot>
        </label>
        <div class="root" part="root">
            ${startSlotTemplate(e,t)}
            <input
                class="control"
                part="control"
                id="control"
                @input="${e=>e.handleTextInput()}"
                @change="${e=>e.handleChange()}"
                ?autofocus="${e=>e.autofocus}"
                ?disabled="${e=>e.disabled}"
                list="${e=>e.list}"
                maxlength="${e=>e.maxlength}"
                minlength="${e=>e.minlength}"
                pattern="${e=>e.pattern}"
                placeholder="${e=>e.placeholder}"
                ?readonly="${e=>e.readOnly}"
                ?required="${e=>e.required}"
                size="${e=>e.size}"
                ?spellcheck="${e=>e.spellcheck}"
                :value="${e=>e.value}"
                type="${e=>e.type}"
                aria-atomic="${e=>e.ariaAtomic}"
                aria-busy="${e=>e.ariaBusy}"
                aria-controls="${e=>e.ariaControls}"
                aria-current="${e=>e.ariaCurrent}"
                aria-describedby="${e=>e.ariaDescribedby}"
                aria-details="${e=>e.ariaDetails}"
                aria-disabled="${e=>e.ariaDisabled}"
                aria-errormessage="${e=>e.ariaErrormessage}"
                aria-flowto="${e=>e.ariaFlowto}"
                aria-haspopup="${e=>e.ariaHaspopup}"
                aria-hidden="${e=>e.ariaHidden}"
                aria-invalid="${e=>e.ariaInvalid}"
                aria-keyshortcuts="${e=>e.ariaKeyshortcuts}"
                aria-label="${e=>e.ariaLabel}"
                aria-labelledby="${e=>e.ariaLabelledby}"
                aria-live="${e=>e.ariaLive}"
                aria-owns="${e=>e.ariaOwns}"
                aria-relevant="${e=>e.ariaRelevant}"
                aria-roledescription="${e=>e.ariaRoledescription}"
                ${ref("control")}
            />
            ${endSlotTemplate(e,t)}
        </div>
    </template>
`,textFieldStyles=(e,t)=>css_css`
	${display("inline-block")} :host {
		font-family: ${fontFamily};
		outline: none;
		user-select: none;
	}
	.root {
		box-sizing: border-box;
		position: relative;
		display: flex;
		flex-direction: row;
		color: ${inputForeground};
		background: ${inputBackground};
		border-radius: calc(${cornerRadius} * 1px);
		border: calc(${borderWidth} * 1px) solid ${dropdownBorder};
		height: calc(${inputHeight} * 1px);
		min-width: ${inputMinWidth};
	}
	.control {
		-webkit-appearance: none;
		font: inherit;
		background: transparent;
		border: 0;
		color: inherit;
		height: calc(100% - (${designUnit} * 1px));
		width: 100%;
		margin-top: auto;
		margin-bottom: auto;
		border: none;
		padding: 0 calc(${designUnit} * 2px + 1px);
		font-size: ${typeRampBaseFontSize};
		line-height: ${typeRampBaseLineHeight};
	}
	.control:hover,
	.control:${focusVisible},
	.control:disabled,
	.control:active {
		outline: none;
	}
	.label {
		display: block;
		color: ${foreground};
		cursor: pointer;
		font-size: ${typeRampBaseFontSize};
		line-height: ${typeRampBaseLineHeight};
		margin-bottom: 2px;
	}
	.label__hidden {
		display: none;
		visibility: hidden;
	}
	.start,
	.end {
		display: flex;
		margin: auto;
		fill: currentcolor;
	}
	::slotted(svg),
	::slotted(span) {
		width: calc(${designUnit} * 4px);
		height: calc(${designUnit} * 4px);
	}
	.start {
		margin-inline-start: calc(${designUnit} * 2px);
	}
	.end {
		margin-inline-end: calc(${designUnit} * 2px);
	}
	:host(:hover:not([disabled])) .root {
		background: ${inputBackground};
		border-color: ${dropdownBorder};
	}
	:host(:active:not([disabled])) .root {
		background: ${inputBackground};
		border-color: ${focusBorder};
	}
	:host(:focus-within:not([disabled])) .root {
		border-color: ${focusBorder};
	}
	:host([disabled]) .label,
	:host([readonly]) .label,
	:host([readonly]) .control,
	:host([disabled]) .control {
		cursor: ${disabledCursor};
	}
	:host([disabled]) {
		opacity: ${disabledOpacity};
	}
	:host([disabled]) .control {
		border-color: ${dropdownBorder};
	}